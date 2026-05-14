"""
Experiment script for MVO with S&P 100 Portfolio.
Uses ANOR-style settings: seq_len=63, pred_len=21, batch training.

Methods:
- mse: MSE loss (two-stage baseline)
- projection: MSE loss with tangent-space projection gradient
- bpqp: Regret loss with BPQP (OSQP-based)
- qpth: Regret loss with QPTH (OptNet)
- cvxpy: Regret loss with CVXPYLayers

Usage:
    python -m experiments.qp.run --method mse --seed 0
    python -m experiments.qp.run --method projection --seed 0
    python -m experiments.qp.run --method bpqp --seed 0
"""

import os
import sys
import json
import argparse
import time
import logging

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'mvo.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from experiments.qp.models import DLinear
from experiments.qp.solvers import create_solver, ProjectionLoss, ProjectionLossTorch, OracleSolver


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0

    def step(self, val: float, epoch: int, model: nn.Module) -> bool:
        if val < self.best_val:
            self.best_val = val
            self.best_epoch = epoch
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def load_data(filepath: str) -> pd.DataFrame:
    """Load S&P 100 returns data (ANOR format)."""
    df = pd.read_csv(filepath)

    # Drop Date column if exists
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Fill missing values
    df = df.ffill().fillna(0.0)

    return df


def prepare_data_tensors(
    returns_df: pd.DataFrame,
    seq_len: int = 63,
    pred_len: int = 21,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> dict:
    """Prepare train/val/test data tensors (ANOR style)."""
    data_values = returns_df.values.astype(np.float32)
    n_assets = data_values.shape[1]

    total_len = len(data_values)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)

    train_data = data_values[:train_len]
    val_data = data_values[train_len:train_len+val_len]
    test_data = data_values[train_len+val_len:]

    def rolling_window(data, seq, pred):
        X, y = [], []
        for i in range(seq, len(data) - pred + 1):
            X.append(data[i-seq:i])
            y.append(data[i:i+pred])
        if len(X) == 0:
            return torch.empty(0, seq, n_assets), torch.empty(0, pred, n_assets)
        return torch.tensor(np.array(X)), torch.tensor(np.array(y))

    X_train, y_train = rolling_window(train_data, seq_len, pred_len)
    X_val, y_val = rolling_window(val_data, seq_len, pred_len)
    X_test, y_test = rolling_window(test_data, seq_len, pred_len)

    test_daily_returns = data_values[train_len+val_len+seq_len:]

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'test_daily_returns': test_daily_returns,
        'n_assets': n_assets
    }


def compute_stats_vectorized(returns_batch, history_batch, epsilon=1e-4):
    """
    Compute mean and covariance from combined history + prediction (ANOR style).

    Args:
        returns_batch: (B, pred_len, n_assets)
        history_batch: (B, seq_len, n_assets)

    Returns:
        mu: (B, n_assets)
        L: Cholesky of covariance (B, n_assets, n_assets)
        cov: (B, n_assets, n_assets)
    """
    B, P, N = returns_batch.shape

    mu = returns_batch.mean(dim=1)  # (B, n_assets)

    # Combined history + prediction for covariance
    combined = torch.cat([history_batch, returns_batch], dim=1)
    centered = combined - combined.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (combined.shape[1] - 1)

    # Spectral shifting for PSD
    eigvals = torch.linalg.eigvalsh(cov)
    min_eigvals = eigvals[:, 0]
    corrections = torch.clamp(-min_eigvals + epsilon, min=0)
    eye = torch.eye(N, device=returns_batch.device).unsqueeze(0)
    cov_safe = cov + corrections.view(-1, 1, 1) * eye

    try:
        L = torch.linalg.cholesky(cov_safe)
    except RuntimeError:
        L = torch.eye(N, device=returns_batch.device).unsqueeze(0).repeat(B, 1, 1)

    return mu, L, cov_safe


def calculate_utility(w, mu, cov, lambda_val):
    """
    Calculate portfolio utility: μ'w - (λ/2) * w'Σw.

    This matches the QP objective: max μ'w - (λ/2) w'Σw
    """
    risk = torch.bmm(w.unsqueeze(1), torch.bmm(cov, w.unsqueeze(2))).squeeze()
    ret = (w * mu).sum(dim=1)
    return ret - 0.5 * lambda_val * risk


def precompute_oracle(X, y, solver, lambda_val, device, batch_size=256):
    """Pre-compute oracle utilities for all samples."""
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    utilities = []

    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Pre-computing oracle", leave=False):
            bx, by = bx.to(device), by.to(device)
            mu_true, L_true, cov_true = compute_stats_vectorized(by, bx)
            w_true = solver(mu_true, L_true)
            util = calculate_utility(w_true, mu_true, cov_true, lambda_val)
            utilities.append(util.cpu())

    return torch.cat(utilities)


def run_experiment(
    method: str,
    seed: int,
    save_dir: str,
    # Data parameters
    data_path: str = None,
    seq_len: int = 63,
    pred_len: int = 21,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    # Training parameters
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10,
    risk_aversion: float = 1.0,
    device: str = 'cuda',
    projection_backend: str = 'qpth'
) -> dict:
    """
    Run S&P 100 experiment with ANOR-style settings.

    Args:
        method: 'mse', 'projection', 'bpqp', 'cvxpy', or 'qpth'
        seed: Random seed
        ...
    """
    # Set seed
    set_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Auto-detect data path
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'sp100_returns.csv')

    # Determine loss type for display
    if method == 'mse':
        loss_display = 'MSE'
    elif method == 'projection':
        loss_display = f'Projection (MSE forward, {projection_backend.upper()} active set)'
    elif method == 'projection_batch':
        loss_display = 'Projection (OSQP forward, batched torch backward)'
    else:
        loss_display = 'Regret'

    # Log experiment start
    logger.info(f"START | method={method}, seed={seed}, lr={lr}, lambda={risk_aversion}")

    print(f"\n{'='*60}")
    print(f"MVO with S&P 100 Portfolio (ANOR style)")
    print(f"Method: {method}, Seed: {seed}")
    print(f"Loss: {loss_display}")
    print(f"seq_len: {seq_len}, pred_len: {pred_len}")
    print(f"risk_aversion (lambda): {risk_aversion}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading S&P 100 Portfolio data...")
    returns_df = load_data(data_path)

    print(f"Data loaded: {len(returns_df)} days, {len(returns_df.columns)} assets")

    # Prepare data tensors
    data = prepare_data_tensors(
        returns_df,
        seq_len=seq_len,
        pred_len=pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    n_assets = data['n_assets']

    print(f"\nData shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")

    # Create model and solver
    model = DLinear(seq_len=seq_len, pred_len=pred_len, n_assets=n_assets).to(device)
    solver = create_solver(method=method, n_assets=n_assets, risk_aversion=risk_aversion).to(device)

    # Create oracle solver (Gurobi-based, consistent across all methods)
    oracle_solver = OracleSolver(n_assets=n_assets, risk_aversion=risk_aversion).to(device)

    print(f"\nModel: DLinear (seq_len={seq_len}, pred_len={pred_len})")

    # Pre-compute oracle utilities (for regret loss and evaluation)
    # Uses OracleSolver for consistency across all methods
    print("\nPre-computing oracle utilities (using Gurobi)...")
    train_oracle = precompute_oracle(X_train, y_train, oracle_solver, risk_aversion, device)
    val_oracle = precompute_oracle(X_val, y_val, oracle_solver, risk_aversion, device)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, train_oracle),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, val_oracle),
        batch_size=batch_size, shuffle=False
    )

    # Determine loss type
    if method == 'mse':
        loss_type = 'MSE'
    elif method in ('projection', 'projection_batch'):
        loss_type = 'Projection'
    else:
        loss_type = 'Regret'

    # Training
    print(f"\nTraining {method} ({loss_type})...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience)
    mse_criterion = nn.MSELoss()

    # Create ProjectionLoss for projection method
    if method == 'projection':
        projection_loss = ProjectionLoss(n_assets, risk_aversion, backend=projection_backend).to(device)
    elif method == 'projection_batch':
        projection_loss = ProjectionLossTorch(n_assets, risk_aversion).to(device)

    history = {'train_loss': [], 'val_loss': []}

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bx, by, b_oracle in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            bx, by, b_oracle = bx.to(device), by.to(device), b_oracle.to(device)

            optimizer.zero_grad()

            pred = model(bx)

            if method == 'mse':
                # Pure MSE loss
                loss = mse_criterion(pred, by)
            elif method in ('projection', 'projection_batch'):
                # Projection: MSE forward, projection gradient backward
                mu_pred, L_pred, _ = compute_stats_vectorized(pred, bx)
                mu_true = by.mean(dim=1)
                loss = projection_loss(mu_pred, mu_true, L_pred)
            else:
                # Regret loss (bpqp, cvxpy, qpth)
                mu_pred, L_pred, _ = compute_stats_vectorized(pred, bx)
                _, _, cov_true = compute_stats_vectorized(by, bx)
                w_pred = solver(mu_pred, L_pred)
                mu_true = by.mean(dim=1)

                util_realized = calculate_utility(w_pred, mu_true, cov_true, risk_aversion)
                loss = torch.abs(b_oracle - util_realized).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_train_loss)

        # Validation (always use regret for early stopping - like shortest path)
        model.eval()
        val_regret_sum = 0.0
        val_oracle_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for bx, by, b_oracle in val_loader:
                bx, by, b_oracle = bx.to(device), by.to(device), b_oracle.to(device)
                pred = model(bx)

                # Always compute regret for validation (all methods)
                # Use oracle_solver (Gurobi) for fair evaluation
                mu_pred, L_pred, _ = compute_stats_vectorized(pred, bx)
                _, _, cov_true = compute_stats_vectorized(by, bx)
                w_pred = oracle_solver(mu_pred, L_pred)
                mu_true = by.mean(dim=1)

                util_realized = calculate_utility(w_pred, mu_true, cov_true, risk_aversion)
                val_regret_sum += (b_oracle - util_realized).sum().item()
                val_oracle_sum += b_oracle.abs().sum().item()
                n_val += bx.shape[0]

        # Normalized regret (like shortest path)
        avg_val_regret = val_regret_sum / (val_oracle_sum + 1e-7)
        history['val_loss'].append(avg_val_regret)

        scheduler.step(avg_val_regret)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_regret={avg_val_regret:.6f}, "
              f"best={early_stopping.best_val:.6f}")

        if early_stopping.step(avg_val_regret, epoch, model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    early_stopping.restore(model)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")

    # Evaluate on test set (Sequential: rebalance every pred_len days)
    print("\nEvaluating on test set (sequential, rebalancing every 21 days)...")
    model.eval()

    # Get raw test data for sequential evaluation
    test_daily_returns = data['test_daily_returns']

    # MSE evaluation (rolling window for fair comparison)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    total_mse = 0.0
    total_mse_norm = 0.0
    n_mse = 0

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred, pred_norm = model(bx, return_normalized=True)
            by_norm = model.normalize_target(by)
            total_mse += ((pred - by) ** 2).mean().item() * bx.shape[0]
            total_mse_norm += ((pred_norm - by_norm) ** 2).mean().item() * bx.shape[0]
            n_mse += bx.shape[0]

    test_mse = total_mse / n_mse
    test_mse_norm = total_mse_norm / n_mse

    # Regret evaluation (sequential: rebalance every pred_len days)
    total_regret = 0.0
    total_oracle = 0.0
    n_rebalances = 0
    idx = seq_len

    with torch.no_grad():
        while idx + pred_len <= len(test_daily_returns):
            # Historical window for prediction
            hist_window = test_daily_returns[idx - seq_len:idx]
            X = torch.tensor(hist_window, dtype=torch.float32).unsqueeze(0).to(device)

            # True future returns
            future_returns = test_daily_returns[idx:idx + pred_len]
            y_true = torch.tensor(future_returns, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict and compute statistics
            pred = model(X)
            mu_pred, L_pred, _ = compute_stats_vectorized(pred, X)
            mu_true, L_true, cov_true = compute_stats_vectorized(y_true, X)

            # Portfolios
            w_pred = oracle_solver(mu_pred, L_pred)
            w_oracle = oracle_solver(mu_true, L_true)

            # Utilities (evaluated on true parameters)
            util_pred = calculate_utility(w_pred, mu_true, cov_true, risk_aversion)
            util_oracle = calculate_utility(w_oracle, mu_true, cov_true, risk_aversion)

            total_regret += (util_oracle - util_pred).item()
            total_oracle += util_oracle.abs().item()
            n_rebalances += 1

            idx += pred_len

    test_regret = total_regret / (total_oracle + 1e-7)

    print(f"\nTest Results:")
    print(f"  MSE (raw): {test_mse:.6f}")
    print(f"  MSE (normalized): {test_mse_norm:.6f}")
    print(f"  Normalized Regret: {test_regret*100:.2f}% ({n_rebalances} rebalances)")

    # Log results
    logger.info(f"DONE  | method={method}, seed={seed}, lambda={risk_aversion} | "
                f"MSE={test_mse:.6f}, MSE_norm={test_mse_norm:.6f}, Regret={test_regret*100:.2f}%, time={train_time:.1f}s")

    # Save results
    results = {
        'method': method,
        'seed': seed,
        'risk_aversion': risk_aversion,
        'train_time': train_time,
        'test_mse': test_mse,
        'test_mse_norm': test_mse_norm,
        'test_regret': test_regret,
        'best_val_loss': early_stopping.best_val,
        'n_epochs_trained': len(history['train_loss']),
        'history': history,
        'config': {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience,
            'n_assets': n_assets,
            'projection_backend': projection_backend if method == 'projection' else None,
        }
    }

    os.makedirs(save_dir, exist_ok=True)
    result_file = os.path.join(
        save_dir,
        f'mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.json'
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    # Save best model checkpoint
    model_file = os.path.join(
        save_dir,
        f'mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.pt'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': results['config'],
        'method': method,
        'seed': seed,
        'risk_aversion': risk_aversion,
        'best_epoch': early_stopping.best_epoch,
        'best_val_loss': early_stopping.best_val,
    }, model_file)
    print(f"Best model saved to: {model_file} (epoch {early_stopping.best_epoch + 1})")

    return results


def main():
    parser = argparse.ArgumentParser(description='MVO with S&P 100 (ANOR style)')

    # Experiment parameters
    parser.add_argument('--method', type=str, required=True,
                        choices=['mse', 'projection', 'projection_batch', 'bpqp', 'cvxpy', 'qpth', 'qpth_seq', 'cvxpy_seq'],
                        help='Method to run (projection_batch uses batched torch backward)')
    parser.add_argument('--projection_backend', type=str, default='qpth',
                        choices=['qpth', 'osqp'],
                        help='Backend for projection method active set detection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results/qp')

    # Data parameters
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=63)
    parser.add_argument('--pred_len', type=int, default=21)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--risk_aversion', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    run_experiment(
        method=args.method,
        seed=args.seed,
        save_dir=args.save_dir,
        data_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        risk_aversion=args.risk_aversion,
        device=args.device,
        projection_backend=args.projection_backend
    )


if __name__ == '__main__':
    main()
