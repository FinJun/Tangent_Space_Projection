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
from experiments.qp.run import load_data, compute_stats_vectorized


def calculate_portfolio_metrics(portfolio_values: np.ndarray, rf: float = 0.0) -> dict:
    df = pd.DataFrame({"value": portfolio_values})
    df['daily_return'] = df['value'].pct_change().fillna(0)

    daily_mean = df['daily_return'].mean()
    daily_std = df['daily_return'].std()

    ann_return = (1 + daily_mean) ** 252 - 1
    ann_std = daily_std * np.sqrt(252)

    sharpe = (ann_return - rf) / ann_std if ann_std > 0 else 0.0

    downside_returns = df[df['daily_return'] < 0]['daily_return']
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino = (ann_return - rf) / downside_std if downside_std > 0 else 0.0

    rolling_max = df['value'].cummax()
    drawdown = df['value'] / rolling_max - 1
    mdd = drawdown.min()

    calmar = ann_return / abs(mdd) if mdd != 0 else 0.0

    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    n_days = len(portfolio_values) - 1

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_std,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "mdd": mdd,
        "calmar_ratio": calmar,
        "final_value": portfolio_values[-1],
        "n_days": n_days,
    }


def run_backtest(
    model,
    solver,
    returns_df: pd.DataFrame,
    seq_len: int = 63,
    pred_len: int = 21,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    transaction_cost: float = 0.001,
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

    portfolio_values = [1.0]
    current_weights = np.ones(n_assets) / n_assets

    all_weights = []
    all_returns = []
    rebalance_dates = []

    idx = seq_len
    while idx + pred_len <= len(test_data):
        hist_window = test_data[idx - seq_len:idx]
        X = torch.tensor(hist_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(X)
            mu_pred, L_pred, _ = compute_stats_vectorized(pred, X)

            try:
                new_weights = solver(mu_pred, L_pred).cpu().numpy().squeeze()
            except Exception:
                new_weights = current_weights.copy()

        new_weights = np.maximum(new_weights, 0)
        weight_sum = new_weights.sum()
        if weight_sum == 0:
            new_weights = np.ones(n_assets) / n_assets
        else:
            new_weights /= weight_sum

        turnover = np.sum(np.abs(new_weights - current_weights))
        tc_paid = turnover * transaction_cost
        portfolio_values[-1] *= (1 - tc_paid)

        current_weights = new_weights
        all_weights.append(current_weights.copy())
        rebalance_dates.append(idx)

        daily_returns = test_data[idx:idx + pred_len]
        period_returns = []

        for day_ret in daily_returns:
            port_ret = np.dot(current_weights, day_ret)
            portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
            period_returns.append(port_ret)

            current_weights = current_weights * (1 + day_ret)
            weight_sum = current_weights.sum()
            if weight_sum > 0:
                current_weights /= weight_sum

        all_returns.extend(period_returns)
        idx += pred_len

    portfolio_values = np.array(portfolio_values)

    return {
        "portfolio_values": portfolio_values,
        "weights": np.array(all_weights),
        "daily_returns": np.array(all_returns),
        "rebalance_dates": rebalance_dates,
    }


def evaluate_portfolio(
    method: str,
    seed: int,
    risk_aversion: float = 1.0,
    lr: float = 0.001,
    model_dir: str = 'results/mvo',
    data_path: str = None,
    seq_len: int = 63,
    pred_len: int = 21,
    transaction_cost: float = 0.001,
    device: str = 'cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model_file = os.path.join(model_dir, f'mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.pt')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device)
    config = checkpoint['config']

    print(f"\n{'='*70}")
    print(f"Portfolio Performance Evaluation")
    print(f"Method: {method}, Seed: {seed}, Lambda: {risk_aversion}")
    print(f"{'='*70}")

    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'sp100_returns.csv')

    returns_df = load_data(data_path)
    n_assets = len(returns_df.columns)

    model = DLinear(
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        n_assets=n_assets
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    solver = OracleSolver(
        n_assets=n_assets,
        risk_aversion=risk_aversion
    ).to(device)

    print(f"\nRunning backtest...")
    backtest_results = run_backtest(
        model=model,
        solver=solver,
        returns_df=returns_df,
        seq_len=seq_len,
        pred_len=pred_len,
        transaction_cost=transaction_cost,
        device=device
    )

    portfolio_values = backtest_results['portfolio_values']
    metrics = calculate_portfolio_metrics(portfolio_values)

    print(f"\n{'='*50}")
    print(f"Portfolio Performance Metrics")
    print(f"{'='*50}")
    print(f"  Total Return:      {metrics['total_return']*100:>10.2f}%")
    print(f"  Annualized Return: {metrics['ann_return']*100:>10.2f}%")
    print(f"  Annualized Vol:    {metrics['ann_volatility']*100:>10.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.3f}")
    print(f"  Max Drawdown:      {metrics['mdd']*100:>10.2f}%")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>10.3f}")
    print(f"  Final Value:       {metrics['final_value']:>10.4f}")
    print(f"  Trading Days:      {metrics['n_days']:>10d}")

    results = {
        'method': method,
        'seed': seed,
        'risk_aversion': risk_aversion,
        'transaction_cost': transaction_cost,
        **metrics
    }

    result_file = os.path.join(
        model_dir,
        f'portfolio_{method}_lambda{risk_aversion}_S{seed}.json'
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    values_file = os.path.join(
        model_dir,
        f'portfolio_values_{method}_lambda{risk_aversion}_S{seed}.csv'
    )
    pd.DataFrame({
        'day': range(len(portfolio_values)),
        'value': portfolio_values
    }).to_csv(values_file, index=False)
    print(f"Portfolio values saved to: {values_file}")

    return results, backtest_results


def compare_methods(
    methods: list,
    seeds: list,
    risk_aversion: float = 1.0,
    lr: float = 0.001,
    model_dir: str = 'results/mvo',
    data_path: str = None,
    transaction_cost: float = 0.001,
    device: str = 'cuda'
):
    all_results = []

    for method in methods:
        for seed in seeds:
            try:
                results, _ = evaluate_portfolio(
                    method=method,
                    seed=seed,
                    risk_aversion=risk_aversion,
                    lr=lr,
                    model_dir=model_dir,
                    data_path=data_path,
                    transaction_cost=transaction_cost,
                    device=device
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error for {method} seed {seed}: {e}")

    if not all_results:
        print("No results to compare")
        return

    df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print(f"Summary: Portfolio Performance Comparison")
    print(f"{'='*80}")

    summary = df.groupby('method').agg({
        'ann_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'mdd': ['mean', 'std'],
        'sortino_ratio': ['mean', 'std'],
    }).round(4)

    print(summary)

    summary_file = os.path.join(model_dir, f'portfolio_comparison_lambda{risk_aversion}.csv')
    summary.to_csv(summary_file)
    print(f"\nSummary saved to: {summary_file}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_dir', type=str, default='results/qp')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--risk_aversion', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--transaction_cost', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compare', action='store_true')

    args = parser.parse_args()

    if args.compare or args.method is None:
        methods = ['mse', 'projection', 'bpqp', 'cvxpy', 'qpth']
        seeds = [0, 1, 2, 3, 4] if args.seed is None else [args.seed]

        compare_methods(
            methods=methods,
            seeds=seeds,
            risk_aversion=args.risk_aversion,
            lr=args.lr,
            model_dir=args.model_dir,
            data_path=args.data_path,
            transaction_cost=args.transaction_cost,
            device=args.device
        )
    else:
        seeds = [args.seed] if args.seed is not None else [0]

        for seed in seeds:
            evaluate_portfolio(
                method=args.method,
                seed=seed,
                risk_aversion=args.risk_aversion,
                lr=args.lr,
                model_dir=args.model_dir,
                data_path=args.data_path,
                transaction_cost=args.transaction_cost,
                device=args.device
            )


if __name__ == '__main__':
    main()
