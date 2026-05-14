"""
MVO Constraint Shift Experiment.

Train with long-only constraint (w >= 0), evaluate with short selling allowed (w >= -lb).
This tests generalization when the feasible region changes at test time.

Usage:
    python -m experiments.qp.eval_constraint_shift --methods projection_batch qpth cvxpy mse
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
import pandas as pd
import osqp
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.qp.models import DLinear
from experiments.qp.run import load_data, compute_stats_vectorized, calculate_utility


def solve_mvo_with_short(mu, cov, risk_aversion, lower_bound=0.0):
    """
    Solve MVO with variable lower bound on weights.

    min  (λ/2) w'Σw - μ'w
    s.t. sum(w) = 1
         w >= lower_bound  (lower_bound=0 for long-only, <0 for short selling)

    Returns: optimal weights
    """
    n = len(mu)

    # OSQP form: min (1/2) x'Px + q'x  s.t. l <= Ax <= u
    # To get (λ/2) w'Σw, we need P = λΣ
    P = risk_aversion * cov
    P = 0.5 * (P + P.T)  # Ensure symmetry
    q = -mu

    # Constraints: sum(w) = 1, w >= lower_bound
    A = np.vstack([np.ones((1, n)), np.eye(n)])
    l = np.hstack([1.0, lower_bound * np.ones(n)])
    u = np.hstack([1.0, np.inf * np.ones(n)])

    P_sparse = sparse.csc_matrix(P)
    A_sparse = sparse.csc_matrix(A)

    prob = osqp.OSQP()
    prob.setup(P_sparse, q, A_sparse, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    res = prob.solve()

    if res.x is not None:
        return res.x
    else:
        # Fallback to uniform
        return np.ones(n) / n


def evaluate_with_short_selling(
    model,
    X_test,
    y_test,
    lower_bound: float,
    risk_aversion: float,
    n_assets: int,
    device: str = 'cuda',
    batch_size: int = 64
):
    """Evaluate model with short selling allowed (w >= lower_bound)."""
    model.eval()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_util_pred = 0
    total_util_oracle = 0
    n_samples = 0

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)

            # Predict
            pred = model(bx)
            mu_pred, L_pred, _ = compute_stats_vectorized(pred, bx)
            mu_true, L_true, cov_true = compute_stats_vectorized(by, bx)

            # Convert to numpy for OSQP
            mu_pred_np = mu_pred.cpu().numpy()
            mu_true_np = mu_true.cpu().numpy()
            cov_pred_np = torch.bmm(L_pred, L_pred.transpose(1, 2)).cpu().numpy()
            cov_true_np = cov_true.cpu().numpy()

            batch_size_actual = bx.shape[0]

            for i in range(batch_size_actual):
                # Solve with predicted parameters
                w_pred = solve_mvo_with_short(
                    mu_pred_np[i], cov_pred_np[i], risk_aversion, lower_bound
                )

                # Solve oracle with true parameters
                w_oracle = solve_mvo_with_short(
                    mu_true_np[i], cov_true_np[i], risk_aversion, lower_bound
                )

                # Calculate utilities (with true parameters)
                w_pred_t = torch.tensor(w_pred, dtype=torch.float32, device=device)
                w_oracle_t = torch.tensor(w_oracle, dtype=torch.float32, device=device)

                util_pred = (mu_true[i] @ w_pred_t -
                            0.5 * risk_aversion * w_pred_t @ cov_true[i] @ w_pred_t)
                util_oracle = (mu_true[i] @ w_oracle_t -
                              0.5 * risk_aversion * w_oracle_t @ cov_true[i] @ w_oracle_t)

                total_util_pred += util_pred.item()
                total_util_oracle += util_oracle.item()
                n_samples += 1

    avg_util_pred = total_util_pred / n_samples
    avg_util_oracle = total_util_oracle / n_samples

    # Regret = 1 - (predicted utility / oracle utility)
    regret = 1 - (avg_util_pred / avg_util_oracle) if avg_util_oracle != 0 else 0

    return {
        'lower_bound': lower_bound,
        'util_pred': avg_util_pred,
        'util_oracle': avg_util_oracle,
        'regret': regret
    }


def run_constraint_shift_experiment(
    method: str,
    seed: int,
    risk_aversion: float,
    lower_bounds: list,
    lr: float = 0.001,
    model_dir: str = 'results/qp',
    data_path: str = None,
    device: str = 'cuda'
):
    """Run constraint shift experiment for a single model."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model_file = os.path.join(model_dir, f'mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.pt')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Load data
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'sp100_returns.csv')

    returns_df = load_data(data_path)
    n_assets = len(returns_df.columns)

    # Prepare data
    data_values = returns_df.values.astype(np.float32)
    seq_len = config['seq_len']
    pred_len = config['pred_len']

    X, y = [], []
    for i in range(len(data_values) - seq_len - pred_len + 1):
        X.append(data_values[i:i+seq_len])
        y.append(data_values[i+seq_len:i+seq_len+pred_len])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    # Split
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)

    X_test = X[n_train + n_val:].to(device)
    y_test = y[n_train + n_val:].to(device)

    # Load model
    model = DLinear(
        seq_len=seq_len,
        pred_len=pred_len,
        n_assets=n_assets
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate with different lower bounds
    results = []
    for lb in tqdm(lower_bounds, desc=f"{method} S{seed}", leave=False):
        result = evaluate_with_short_selling(
            model, X_test, y_test, lb, risk_aversion, n_assets, device
        )
        result['method'] = method
        result['seed'] = seed
        result['risk_aversion'] = risk_aversion
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='MVO Constraint Shift Experiment')
    parser.add_argument('--methods', nargs='+', default=['projection_batch', 'qpth', 'cvxpy', 'mse'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--risk_aversion', type=float, default=2.0)
    parser.add_argument('--lower_bounds', nargs='+', type=float,
                        default=[0.0, -0.05, -0.1, -0.2, -0.3])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default='results/qp')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    all_results = []

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for seed in args.seeds:
            try:
                results = run_constraint_shift_experiment(
                    method=method,
                    seed=seed,
                    risk_aversion=args.risk_aversion,
                    lower_bounds=args.lower_bounds,
                    lr=args.lr,
                    model_dir=args.model_dir,
                    device=args.device
                )
                all_results.extend(results)
                print(f"  Seed {seed}: OK")
            except Exception as e:
                print(f"  Seed {seed}: Error - {e}")

    if not all_results:
        print("No results")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Summary by method and lower_bound
    print(f"\n{'='*80}")
    print("Constraint Shift Results: Long-only (lb=0) → Short Allowed (lb<0)")
    print(f"{'='*80}")

    # Print mean regret
    print("\nMean Regret by Lower Bound:")
    mean_df = df.pivot_table(values='regret', index='method', columns='lower_bound', aggfunc='mean')
    print(mean_df.round(4).to_string())

    print("\nStd Regret by Lower Bound:")
    std_df = df.pivot_table(values='regret', index='method', columns='lower_bound', aggfunc='std')
    print(std_df.round(4).to_string())

    # Save results
    output_file = os.path.join(args.model_dir, f'constraint_shift_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save summary
    summary_file = os.path.join(args.model_dir, f'constraint_shift_summary.csv')
    mean_df.to_csv(summary_file)
    print(f"Summary saved to: {summary_file}")

    return df


if __name__ == '__main__':
    main()
