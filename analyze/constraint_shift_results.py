#!/usr/bin/env python
"""
Constraint Shift Experiment Results Analysis

Analyzes:
1. Capacity Generalization (Knapsack) - different capacity ratios
2. Direction Generalization (Shortest Path) - forward vs cross direction
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict


def load_capacity_results(results_dir, degrees, methods, seeds):
    """Load capacity generalization results."""
    data = {}

    for deg in degrees:
        data[deg] = {}
        for method in methods:
            results = {cap: [] for cap in ['0.3', '0.5', '0.7', '0.9']}

            for seed in seeds:
                fpath = os.path.join(results_dir, f"capacity_{method}_d{deg}_seed{seed}.json")
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        d = json.load(f)
                        for cap in ['0.3', '0.5', '0.7', '0.9']:
                            if cap in d['regrets']:
                                results[cap].append(d['regrets'][cap] * 100)

            data[deg][method] = results

    return data


def load_direction_results(results_dir, degrees, methods, seeds):
    """Load direction generalization results."""
    data = {}

    for deg in degrees:
        data[deg] = {}
        for method in methods:
            results = {'forward': [], 'cross': []}

            for seed in seeds:
                fpath = os.path.join(results_dir, f"direction_{method}_d{deg}_seed{seed}.json")
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        d = json.load(f)
                        results['forward'].append(d['regret_forward'] * 100)
                        results['cross'].append(d['regret_shift'] * 100)

            data[deg][method] = results

    return data


def format_mean_std(values, precision=2):
    """Format as mean ± std."""
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    width = 5 + precision
    return f"{mean:>{width}.{precision}f}±{std:<{width}.{precision}f}"


def print_capacity_results(data, methods, precision=2):
    """Print capacity generalization results."""
    col_width = 12 + precision
    print("=" * (12 + col_width * 4 + 4))
    print("CAPACITY GENERALIZATION (Knapsack) - Regret (%)")
    print("=" * (12 + col_width * 4 + 4))
    print()
    print("Training capacity: 0.5")
    print("Test capacities: 0.3, 0.7, 0.9")
    print()

    for deg in sorted(data.keys()):
        print(f"[Degree {deg}]")
        print("-" * (12 + col_width * 4 + 4))
        print(f"{'Method':<12} {'cap=0.3':>{col_width}} {'cap=0.5*':>{col_width}} {'cap=0.7':>{col_width}} {'cap=0.9':>{col_width}}")
        print("-" * (12 + col_width * 4 + 4))

        for method in methods:
            if method not in data[deg]:
                continue
            results = data[deg][method]
            row = f"{method:<12}"
            for cap in ['0.3', '0.5', '0.7', '0.9']:
                row += f" {format_mean_std(results[cap], precision):>{col_width}}"
            print(row)
        print()


def print_direction_results(data, methods, precision=2):
    """Print direction generalization results."""
    col_width = 12 + precision
    print("=" * (12 + col_width * 2 + 14 + 3))
    print("DIRECTION GENERALIZATION (Shortest Path) - Regret (%)")
    print("=" * (12 + col_width * 2 + 14 + 3))
    print()
    print("Training direction: Forward (top-left -> bottom-right)")
    print("Test direction: Cross (top-right -> bottom-left)")
    print()

    for deg in sorted(data.keys()):
        print(f"[Degree {deg}]")
        print("-" * (12 + col_width * 2 + 14 + 3))
        print(f"{'Method':<12} {'Forward*':>{col_width}} {'Cross':>{col_width}} {'Δ Regret':>14}")
        print("-" * (12 + col_width * 2 + 14 + 3))

        for method in methods:
            if method not in data[deg]:
                continue
            results = data[deg][method]

            fwd = results['forward']
            cross = results['cross']

            if fwd and cross:
                fwd_str = format_mean_std(fwd, precision)
                cross_str = format_mean_std(cross, precision)
                delta = np.mean(cross) - np.mean(fwd)
                delta_str = f"{delta:>+7.{precision}f}"
            else:
                fwd_str = cross_str = delta_str = "N/A"

            print(f"{method:<12} {fwd_str:>{col_width}} {cross_str:>{col_width}} {delta_str:>14}")
        print()


def print_summary_table(cap_data, dir_data, methods, deg=8):
    """Print summary table for a specific degree."""
    print("=" * 80)
    print(f"SUMMARY TABLE (Degree {deg})")
    print("=" * 80)
    print()

    # Capacity summary (train vs avg shifted)
    print("Capacity Generalization (KS):")
    print("-" * 60)
    print(f"{'Method':<12} {'Train (0.5)':>14} {'Shifted (avg)':>14} {'Δ':>10}")
    print("-" * 60)

    for method in methods:
        if method not in cap_data[deg]:
            continue
        results = cap_data[deg][method]

        train = results['0.5']
        shifted = results['0.3'] + results['0.7'] + results['0.9']

        if train and shifted:
            train_mean = np.mean(train)
            shifted_mean = np.mean(shifted)
            delta = shifted_mean - train_mean
            print(f"{method:<12} {train_mean:>10.2f}%    {shifted_mean:>10.2f}%    {delta:>+8.2f}")

    print()

    # Direction summary
    print("Direction Generalization (SP):")
    print("-" * 60)
    print(f"{'Method':<12} {'Forward':>14} {'Cross':>14} {'Δ':>10}")
    print("-" * 60)

    for method in methods:
        if method not in dir_data[deg]:
            continue
        results = dir_data[deg][method]

        fwd = results['forward']
        cross = results['cross']

        if fwd and cross:
            fwd_mean = np.mean(fwd)
            cross_mean = np.mean(cross)
            delta = cross_mean - fwd_mean
            print(f"{method:<12} {fwd_mean:>10.2f}%    {cross_mean:>10.2f}%    {delta:>+8.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze constraint shift experiment results")
    parser.add_argument('--results_dir', type=str, default='results/constraint_shift',
                        help='Directory containing result JSON files')
    parser.add_argument('--degrees', type=int, nargs='+', default=[2, 4, 6, 8],
                        help='Degrees to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['mse', 'spo', 'dbb', 'pfyl', 'projection', 'lava'],
                        help='Methods to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seeds to include')
    parser.add_argument('--exclude_seeds', type=int, nargs='+', default=[],
                        help='Seeds to exclude (e.g., outliers)')
    parser.add_argument('--summary_deg', type=int, default=8,
                        help='Degree for summary table')
    parser.add_argument('--capacity_only', action='store_true',
                        help='Show only capacity results')
    parser.add_argument('--direction_only', action='store_true',
                        help='Show only direction results')
    parser.add_argument('--precision', type=int, default=2,
                        help='Decimal places for results (default: 2)')
    args = parser.parse_args()

    # Filter seeds
    seeds = [s for s in args.seeds if s not in args.exclude_seeds]

    print(f"Results directory: {args.results_dir}")
    print(f"Degrees: {args.degrees}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {seeds}")
    if args.exclude_seeds:
        print(f"Excluded seeds: {args.exclude_seeds}")
    print()

    # Load results
    cap_data = load_capacity_results(args.results_dir, args.degrees, args.methods, seeds)
    dir_data = load_direction_results(args.results_dir, args.degrees, args.methods, seeds)

    # Print results
    if not args.direction_only:
        print_capacity_results(cap_data, args.methods, args.precision)

    if not args.capacity_only:
        print_direction_results(dir_data, args.methods, args.precision)

    # Summary
    if args.summary_deg in args.degrees:
        print_summary_table(cap_data, dir_data, args.methods, args.summary_deg)


if __name__ == "__main__":
    main()
