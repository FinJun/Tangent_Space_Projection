import os
import sys
import argparse
import json

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.qp.models import DLinear
from experiments.qp.solvers import OracleSolver
from experiments.qp.run import load_data, compute_stats_vectorized, calculate_utility


def run_sequential_regret_evaluation(
    model,
    oracle_solver,
    returns_df: pd.DataFrame,
    risk_aversion: float,
    seq_len: int = 63,
    pred_len: int = 21,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    device: str = 'cuda'
) -> dict:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()

    data_values = returns_df.values.astype(np.float32)
    n_assets = data_values.shape[1]

    total_len = len(data_values)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_start = train_len + val_len

    test_data = data_values[test_start:]

    all_regrets = []
    all_oracle_utils = []
    all_realized_utils = []
    rebalance_dates = []

    idx = seq_len

    while idx + pred_len <= len(test_data):
        hist_window = test_data[idx - seq_len:idx]
        X = torch.tensor(hist_window, dtype=torch.float32).unsqueeze(0).to(device)

        future_returns = test_data[idx:idx + pred_len]
        y_true = torch.tensor(future_returns, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(X)

            mu_pred, L_pred, cov_pred = compute_stats_vectorized(pred, X)
            mu_true, L_true, cov_true = compute_stats_vectorized(y_true, X)

            w_pred = oracle_solver(mu_pred, L_pred)
            w_oracle = oracle_solver(mu_true, L_true)

            util_realized = calculate_utility(w_pred, mu_true, cov_true, risk_aversion)
            util_oracle = calculate_utility(w_oracle, mu_true, cov_true, risk_aversion)

            regret = (util_oracle - util_realized).item()

            all_regrets.append(regret)
            all_oracle_utils.append(util_oracle.item())
            all_realized_utils.append(util_realized.item())
            rebalance_dates.append(idx)

        idx += pred_len

    all_regrets = np.array(all_regrets)
    all_oracle_utils = np.array(all_oracle_utils)
    all_realized_utils = np.array(all_realized_utils)

    total_regret = all_regrets.sum()
    total_oracle = np.abs(all_oracle_utils).sum()
    normalized_regret = total_regret / (total_oracle + 1e-7)

    mean_regret = all_regrets.mean()
    std_regret = all_regrets.std()

    return {
        'normalized_regret': normalized_regret,
        'mean_regret': mean_regret,
        'std_regret': std_regret,
        'total_regret': total_regret,
        'total_oracle_utility': total_oracle,
        'n_rebalances': len(all_regrets),
        'all_regrets': all_regrets.tolist(),
        'all_oracle_utils': all_oracle_utils.tolist(),
        'all_realized_utils': all_realized_utils.tolist(),
    }


def evaluate_regret(
    method: str,
    seed: int,
    risk_aversion: float = 1.0,
    lr: float = 0.001,
    model_dir: str = 'results/qp',
    data_path: str = None,
    seq_len: int = 63,
    pred_len: int = 21,
    device: str = 'cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model_file = os.path.join(model_dir, f'mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.pt')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    config = checkpoint['config']

    print(f"\n{'='*70}")
    print(f"Sequential Test Regret Evaluation (21-day rebalancing)")
    print(f"Method: {method}, Seed: {seed}, Lambda: {risk_aversion}")
    print(f"{'='*70}")

    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'sp100_returns.csv')

    returns_df = load_data(data_path)
    n_assets = len(returns_df.columns)

    print(f"Data: {len(returns_df)} days, {n_assets} assets")

    model = DLinear(
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        n_assets=n_assets
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    oracle_solver = OracleSolver(n_assets=n_assets, risk_aversion=risk_aversion).to(device)

    print(f"\nRunning sequential evaluation (rebalancing every {pred_len} days)...")
    results = run_sequential_regret_evaluation(
        model=model,
        oracle_solver=oracle_solver,
        returns_df=returns_df,
        risk_aversion=risk_aversion,
        seq_len=seq_len,
        pred_len=pred_len,
        device=device
    )

    print(f"\n{'='*50}")
    print(f"Sequential Test Regret Results")
    print(f"{'='*50}")
    print(f"  Normalized Regret:     {results['normalized_regret']*100:>10.4f}%")
    print(f"  Mean Regret:           {results['mean_regret']:>10.6f}")
    print(f"  Std Regret:            {results['std_regret']:>10.6f}")
    print(f"  Total Regret:          {results['total_regret']:>10.6f}")
    print(f"  Total Oracle Utility:  {results['total_oracle_utility']:>10.6f}")
    print(f"  Number of Rebalances:  {results['n_rebalances']:>10d}")

    save_results = {
        'method': method,
        'seed': seed,
        'risk_aversion': risk_aversion,
        'normalized_regret': results['normalized_regret'],
        'mean_regret': results['mean_regret'],
        'std_regret': results['std_regret'],
        'total_regret': results['total_regret'],
        'total_oracle_utility': results['total_oracle_utility'],
        'n_rebalances': results['n_rebalances'],
    }

    result_file = os.path.join(
        model_dir,
        f'sequential_regret_{method}_lambda{risk_aversion}_S{seed}.json'
    )
    with open(result_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    return results


def compare_methods(
    methods: list,
    seeds: list,
    risk_aversion: float = 1.0,
    lr: float = 0.001,
    model_dir: str = 'results/qp',
    data_path: str = None,
    device: str = 'cuda'
):
    all_results = []

    for method in methods:
        for seed in seeds:
            try:
                results = evaluate_regret(
                    method=method,
                    seed=seed,
                    risk_aversion=risk_aversion,
                    lr=lr,
                    model_dir=model_dir,
                    data_path=data_path,
                    device=device
                )
                all_results.append({
                    'method': method,
                    'seed': seed,
                    'normalized_regret': results['normalized_regret'],
                    'mean_regret': results['mean_regret'],
                    'std_regret': results['std_regret'],
                    'n_rebalances': results['n_rebalances'],
                })
            except Exception as e:
                print(f"Error for {method} seed {seed}: {e}")

    if not all_results:
        print("No results to compare")
        return

    df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print(f"Summary: Sequential Test Regret Comparison (21-day rebalancing)")
    print(f"{'='*80}")

    summary = df.groupby('method').agg({
        'normalized_regret': ['mean', 'std'],
        'mean_regret': ['mean', 'std'],
    })

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'normalized_regret_mean': 'Norm. Regret (mean)',
        'normalized_regret_std': 'Norm. Regret (std)',
        'mean_regret_mean': 'Mean Regret (mean)',
        'mean_regret_std': 'Mean Regret (std)',
    })

    print(f"\n{'Method':<20} {'Norm. Regret':<25} {'Mean Regret per Period':<25}")
    print(f"{'-'*70}")

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        nr_mean = method_df['normalized_regret'].mean() * 100
        nr_std = method_df['normalized_regret'].std() * 100
        mr_mean = method_df['mean_regret'].mean()
        mr_std = method_df['mean_regret'].std()
        print(f"{method:<20} {nr_mean:>8.4f}% +/- {nr_std:>6.4f}%    {mr_mean:>10.6f} +/- {mr_std:>8.6f}")

    summary_file = os.path.join(model_dir, f'sequential_regret_comparison_lambda{risk_aversion}.csv')
    df.to_csv(summary_file, index=False)
    print(f"\nDetailed results saved to: {summary_file}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_dir', type=str, default='results/qp')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--risk_aversion', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compare', action='store_true')

    args = parser.parse_args()

    if args.compare or args.method is None:
        methods = ['mse', 'projection', 'projection_batch', 'bpqp', 'cvxpy', 'qpth']
        seeds = [0, 1, 2, 3, 4] if args.seed is None else [args.seed]

        compare_methods(
            methods=methods,
            seeds=seeds,
            risk_aversion=args.risk_aversion,
            lr=args.lr,
            model_dir=args.model_dir,
            data_path=args.data_path,
            device=args.device
        )
    else:
        seeds = [args.seed] if args.seed is not None else [0]

        for seed in seeds:
            evaluate_regret(
                method=args.method,
                seed=seed,
                risk_aversion=args.risk_aversion,
                lr=args.lr,
                model_dir=args.model_dir,
                data_path=args.data_path,
                device=args.device
            )


if __name__ == '__main__':
    main()
