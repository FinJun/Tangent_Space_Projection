#!/usr/bin/env python
"""
MVO (Mean-Variance Optimization) Experiment Results Analysis

Analyzes:
1. Main MVO results (regret, MSE, time)
2. Constraint shift results (lower bound generalization)
3. Portfolio performance metrics
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob


def load_mvo_results(results_dir, methods, seeds, risk_aversion=2.0, lr=0.001):
    """Load MVO experiment results from JSON files."""
    data = {}

    for method in methods:
        regrets = []
        mses = []
        times = []

        for seed in seeds:
            pattern = f"{results_dir}/mvo_{method}_lambda{risk_aversion}_lr{lr}_S{seed}.json"
            if os.path.exists(pattern):
                with open(pattern) as f:
                    d = json.load(f)
                    regrets.append(d.get('test_regret', 0) * 100)
                    mses.append(d.get('test_mse', 0))
                    times.append(d.get('train_time', 0))

        data[method] = {
            'regret': regrets,
            'mse': mses,
            'time': times
        }

    return data


def load_constraint_shift_results(results_dir):
    """Load constraint shift results from CSV."""
    csv_path = f"{results_dir}/constraint_shift_results.csv"
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def load_portfolio_results(results_dir):
    """Load portfolio performance results from CSV."""
    csv_path = f"{results_dir}/portfolio_constraint_shift.csv"
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def format_mean_std(values, precision=2):
    """Format as mean±std."""
    if not values or len(values) == 0:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def print_main_results(data, methods, precision=2):
    """Print main MVO results."""
    print("=" * 70)
    print("MVO Main Results")
    print("=" * 70)

    col_width = 14 + precision

    # Header
    print(f"{'Method':<15} {'Regret (%)':>{col_width}} {'MSE':>{col_width}} {'Time (s)':>{col_width}}")
    print("-" * 70)

    for method in methods:
        d = data.get(method, {})
        regret = format_mean_std(d.get('regret', []), precision)
        mse = format_mean_std(d.get('mse', []), precision+2)
        time = format_mean_std(d.get('time', []), 1)
        print(f"{method:<15} {regret:>{col_width}} {mse:>{col_width}} {time:>{col_width}}")


def print_constraint_shift_results(df, methods, precision=2):
    """Print constraint shift (lower bound generalization) results."""
    if df is None:
        print("No constraint shift results found.")
        return

    print("\n" + "=" * 70)
    print("MVO Constraint Shift (Lower Bound Generalization) - Regret")
    print("=" * 70)
    print("\nTraining lower bound: 0.0")
    print("Test lower bounds: -0.1, -0.3, -0.5, -1.0\n")

    col_width = 12 + precision

    # Header
    header = f"{'Method':<15}"
    for lb in [-0.1, -0.3, -0.5, -1.0]:
        header += f" {'lb='+str(lb):>{col_width}}"
    print(header)
    print("-" * len(header))

    method_map = {
        'mse': 'mse',
        'qpth': 'qpth',
        'cvxpy': 'cvxpy',
        'projection': 'projection_batch',
        'projection_batch': 'projection_batch'
    }

    for method in methods:
        method_key = method_map.get(method, method)
        row = f"{method:<15}"

        for lb in [-0.1, -0.3, -0.5, -1.0]:
            subset = df[(df['method'] == method_key) & (df['lower_bound'] == lb)]
            if len(subset) > 0:
                mean = subset['regret'].mean()
                std = subset['regret'].std()
                row += f" {mean:.{precision}f}±{std:.{precision}f}".rjust(col_width + 1)
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)


def print_portfolio_performance(df, methods, precision=2):
    """Print portfolio performance results."""
    if df is None:
        print("No portfolio performance results found.")
        return

    print("\n" + "=" * 70)
    print("Portfolio Performance (lb=0.0)")
    print("=" * 70)

    col_width = 14 + precision

    # Filter lb=0.0
    df_lb0 = df[df['lb'] == 0.0]

    if len(df_lb0) == 0:
        print("No lb=0.0 results found.")
        return

    # Header
    print(f"{'Method':<15} {'Ann. Return':>{col_width}} {'Sharpe':>{col_width}} {'MDD':>{col_width}}")
    print("-" * 70)

    method_map = {
        'mse': 'mse',
        'qpth': 'qpth',
        'cvxpy': 'cvxpy',
        'projection': 'projection_batch',
        'projection_batch': 'projection_batch'
    }

    for method in methods:
        method_key = method_map.get(method, method)
        subset = df_lb0[df_lb0['method'] == method_key]

        if len(subset) > 0:
            row = subset.iloc[0]
            ann_ret = f"{row['ann_return_mean']:.{precision}f}±{row['ann_return_std']:.{precision}f}"
            sharpe = f"{row['sharpe_mean']:.{precision}f}±{row['sharpe_std']:.{precision}f}"
            mdd = f"{row['mdd_mean']:.{precision}f}±{row['mdd_std']:.{precision}f}"
            print(f"{method:<15} {ann_ret:>{col_width}} {sharpe:>{col_width}} {mdd:>{col_width}}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MVO experiment results")
    parser.add_argument('--results_dir', type=str, default='results/qp',
                        help='Directory containing result files')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['mse', 'qpth', 'cvxpy', 'projection'],
                        help='Methods to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seeds to include')
    parser.add_argument('--precision', type=int, default=2,
                        help='Decimal places for results')
    parser.add_argument('--risk_aversion', type=float, default=2.0,
                        help='Risk aversion parameter')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--constraint_shift', action='store_true',
                        help='Show constraint shift results')
    parser.add_argument('--portfolio', action='store_true',
                        help='Show portfolio performance results')
    parser.add_argument('--all', action='store_true',
                        help='Show all results')
    args = parser.parse_args()

    # Map method names for JSON files
    json_methods = []
    for m in args.methods:
        if m == 'projection':
            json_methods.append('projection_batch')
        else:
            json_methods.append(m)

    # Load and print main results
    if args.all or (not args.constraint_shift and not args.portfolio):
        data = load_mvo_results(args.results_dir, json_methods, args.seeds,
                                args.risk_aversion, args.lr)
        # Rename back for display
        display_data = {}
        for m in args.methods:
            key = 'projection_batch' if m == 'projection' else m
            display_data[m] = data.get(key, {})
        print_main_results(display_data, args.methods, args.precision)

    # Load and print constraint shift results
    if args.all or args.constraint_shift:
        df = load_constraint_shift_results(args.results_dir)
        print_constraint_shift_results(df, args.methods, args.precision)

    # Load and print portfolio performance
    if args.all or args.portfolio:
        df = load_portfolio_results(args.results_dir)
        print_portfolio_performance(df, args.methods, args.precision)


if __name__ == "__main__":
    main()
