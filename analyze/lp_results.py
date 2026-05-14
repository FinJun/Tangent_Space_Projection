#!/usr/bin/env python
"""
LP Experiment Results Analysis

Analyzes:
1. Shortest Path (SP) results
2. Knapsack (KS) results
3. Noise experiments
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob


def load_lp_results(results_dir, prob, methods, degrees, seeds, noise=0.0):
    """Load LP experiment results from CSV files."""
    data = {}

    for deg in degrees:
        data[deg] = {}
        for method in methods:
            regrets = []
            mses = []
            times = []

            for seed in seeds:
                # Build filename
                if noise > 0:
                    pattern = f"{results_dir}/{prob}_{method}_d{deg}_noise{noise}_seed{seed}.csv"
                else:
                    pattern = f"{results_dir}/{prob}_{method}_d{deg}_seed{seed}.csv"

                files = glob(pattern)
                if files:
                    df = pd.read_csv(files[0])
                    regrets.append(df['regret'].iloc[-1] * 100)
                    mses.append(df['mse'].iloc[-1])
                    if 'time' in df.columns:
                        times.append(df['time'].iloc[-1])

            data[deg][method] = {
                'regret': regrets,
                'mse': mses,
                'time': times
            }

    return data


def format_mean_std(values, precision=2):
    """Format as mean±std."""
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def print_results(data, methods, metric='regret', precision=2, title=""):
    """Print results table."""
    if title:
        print(f"\n{'='*70}")
        print(title)
        print('='*70)

    col_width = 10 + precision * 2

    # Header
    header = f"{'Deg':<6}"
    for method in methods:
        header += f" {method:>{col_width}}"
    print(header)
    print("-" * len(header))

    for deg in sorted(data.keys()):
        row = f"{deg:<6}"
        for method in methods:
            vals = data[deg].get(method, {}).get(metric, [])
            row += f" {format_mean_std(vals, precision):>{col_width}}"
        print(row)


def print_noise_comparison(results_dir, prob, methods, deg, noises, seeds, precision=2):
    """Print noise comparison table."""
    print(f"\n{'='*70}")
    print(f"{prob.upper()} Noise Comparison (deg={deg}) - Regret (%)")
    print('='*70)

    col_width = 12 + precision

    # Header
    header = f"{'Method':<12}"
    for noise in noises:
        header += f" {'noise='+str(noise):>{col_width}}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<12}"
        for noise in noises:
            data = load_lp_results(results_dir, prob, [method], [deg], seeds, noise)
            vals = data[deg].get(method, {}).get('regret', [])
            row += f" {format_mean_std(vals, precision):>{col_width}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Analyze LP experiment results")
    parser.add_argument('--results_dir', type=str, default='results/lp',
                        help='Directory containing result CSV files')
    parser.add_argument('--noise_dir', type=str, default='results/lp_noise',
                        help='Directory containing noise experiment results')
    parser.add_argument('--prob', type=str, choices=['sp', 'ks', 'both'], default='both',
                        help='Problem type (sp, ks, or both)')
    parser.add_argument('--degrees', type=int, nargs='+', default=[2, 4, 6, 8],
                        help='Degrees to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['mse', 'spo', 'dbb', 'pfyl', 'projection', 'lava'],
                        help='Methods to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seeds to include')
    parser.add_argument('--precision', type=int, default=2,
                        help='Decimal places for results')
    parser.add_argument('--metric', type=str, choices=['regret', 'mse', 'time'], default='regret',
                        help='Metric to display')
    parser.add_argument('--noise', action='store_true',
                        help='Show noise experiment results')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.1, 0.3, 0.5],
                        help='Noise levels to compare')
    parser.add_argument('--noise_deg', type=int, default=8,
                        help='Degree for noise comparison')
    args = parser.parse_args()

    probs = ['sp', 'ks'] if args.prob == 'both' else [args.prob]

    if args.noise:
        # Noise comparison
        for prob in probs:
            print_noise_comparison(
                args.noise_dir, prob, args.methods, args.noise_deg,
                args.noise_levels, args.seeds, args.precision
            )
    else:
        # Standard results
        metric_name = {'regret': 'Regret (%)', 'mse': 'MSE', 'time': 'Time (s)'}

        for prob in probs:
            data = load_lp_results(args.results_dir, prob, args.methods, args.degrees, args.seeds)
            prob_name = 'Shortest Path' if prob == 'sp' else 'Knapsack'
            print_results(
                data, args.methods, args.metric, args.precision,
                f"{prob_name} - {metric_name[args.metric]}"
            )


if __name__ == "__main__":
    main()
