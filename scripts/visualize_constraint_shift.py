#!/usr/bin/env python
"""Visualize constraint shift experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import glob
import os

# Publication-Quality Style
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

palette = {
    'MSE': '#A0A0A0',
    'SPO+': '#5B7C99',
    'DBB': '#7A9A7A',
    'PFYL': '#8B7B96',
    'LAVA': '#D4A574',
    'Projection': '#B8433E',
    'Projection Dual': '#5A3E22',
}

METHOD_MAP = {
    'mse': 'MSE',
    'spo': 'SPO+',
    'dbb': 'DBB',
    'pfyl': 'PFYL',
    'lava': 'LAVA',
    'projection': 'Projection',
    'projection-dual': 'Projection Dual',
    'projection-dual-scaled': 'Projection Dual',
    'projection_batch': 'Projection',
    'qpth': 'QPTH',
    'cvxpy': 'CvxpyLayers',
    'bpqp': 'BPQP'
}

# Extended palette for MVO methods
palette_mvo = {
    'MSE': '#A0A0A0',
    'Projection': '#B8433E',
    'QPTH': '#5B7C99',
    'CvxpyLayers': '#7A9A7A',
    'BPQP': '#8B7B96',
}


def load_capacity_data(results_dir='results/constraint_shift'):
    """Load capacity generalization results."""
    data = []
    for json_file in glob.glob(os.path.join(results_dir, 'capacity_*.json')):
        with open(json_file, 'r') as f:
            result = json.load(f)
            method = METHOD_MAP.get(result['method'], result['method'])
            if 'results' in result:
                for cap_result in result['results']:
                    data.append({
                        'Method': method,
                        'Capacity': cap_result['capacity_ratio'],
                        'Regret (%)': cap_result['regret'] * 100,
                        'MSE': cap_result['mse'],
                        'seed': result['seed']
                    })
            else:
                ratios = result.get('capacity_ratios', [])
                regrets = result.get('regrets', {})
                mses = result.get('mses', {})
                for ratio in ratios:
                    data.append({
                        'Method': method,
                        'Capacity': ratio,
                        'Regret (%)': float(regrets.get(str(ratio), 0.0)) * 100,
                        'MSE': float(mses.get(str(ratio), 0.0)),
                        'seed': result['seed']
                    })

    if not data:
        return None
    return pd.DataFrame(data)


def load_direction_data(results_dir='results/constraint_shift'):
    """Load direction generalization results."""
    data = []
    for json_file in glob.glob(os.path.join(results_dir, 'direction_*.json')):
        with open(json_file, 'r') as f:
            result = json.load(f)
            method = METHOD_MAP.get(result['method'], result['method'])
            forward = result.get('regret_forward') or result.get('regret')
            reverse = result.get('regret_reverse') or result.get('regret_shift') or result.get('regret_cross')
            if forward is not None:
                data.append({
                    'Method': method,
                    'Direction': 'Forward (Train)',
                    'Regret (%)': forward * 100,
                    'seed': result['seed']
                })
            if 'regret_cross' in result or 'regret_shift' in result:
                cross_val = result.get('regret_cross') or result.get('regret_shift')
                data.append({
                    'Method': method,
                    'Direction': 'Cross (Test)',
                    'Regret (%)': cross_val * 100,
                    'seed': result['seed']
                })
            elif reverse is not None:
                data.append({
                    'Method': method,
                    'Direction': 'Reverse (Test)',
                    'Regret (%)': reverse * 100,
                    'seed': result['seed']
                })

    if not data:
        return None
    return pd.DataFrame(data)


def plot_capacity_generalization(df, save_dir='figures'):
    """Plot capacity generalization results."""
    os.makedirs(save_dir, exist_ok=True)

    method_order = ['MSE', 'SPO+', 'DBB', 'PFYL', 'LAVA', 'Projection', 'Projection Dual']
    method_order = [m for m in method_order if m in df['Method'].unique()]
    capacities = sorted(df['Capacity'].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    markers = {'MSE': 'o', 'SPO+': 's', 'DBB': '^', 'PFYL': 'D', 'LAVA': 'v', 'Projection': 'P',
               'Projection Dual': 'X'}

    for method in method_order:
        method_data = df[df['Method'] == method]
        means = method_data.groupby('Capacity')['Regret (%)'].mean()
        stds = method_data.groupby('Capacity')['Regret (%)'].std()

        lw = 2.5 if method == 'Projection' else 1.5
        ms = 10 if method == 'Projection' else 7

        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=method, color=palette[method], marker=markers[method],
                    capsize=4, linewidth=lw, markersize=ms,
                    zorder=5 if method == 'Projection' else 3)

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Train Capacity')
    ax.set_xlabel('Capacity Ratio')
    ax.set_ylabel('Regret (%)')
    ax.set_title('Knapsack: Capacity Generalization', fontweight='bold', pad=12)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'constraint_shift_capacity.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, 'constraint_shift_capacity.pdf'), facecolor='white')
    print(f"Saved {save_dir}/constraint_shift_capacity.png & .pdf")
    plt.close()


def plot_direction_generalization(df, save_dir='figures'):
    """Plot direction generalization results."""
    os.makedirs(save_dir, exist_ok=True)

    method_order = ['MSE', 'SPO+', 'DBB', 'PFYL', 'LAVA', 'Projection']
    method_order = [m for m in method_order if m in df['Method'].unique()]
    direction_order = ['Forward (Train)', 'Cross (Test)', 'Reverse (Test)']

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(data=df, x='Direction', y='Regret (%)', hue='Method',
                palette=palette, capsize=.05, order=direction_order,
                hue_order=method_order, errorbar='sd', ax=ax,
                err_kws={'linewidth': 1.2}, edgecolor='white', linewidth=1)

    ax.set_title("Shortest Path: Direction Generalization", fontweight='bold', pad=12)
    ax.set_ylabel("Regret (%)")
    ax.set_xlabel("")

    leg = ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left',
                    frameon=True, fancybox=False, edgecolor='#CCCCCC')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'constraint_shift_direction.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, 'constraint_shift_direction.pdf'), facecolor='white')
    print(f"Saved {save_dir}/constraint_shift_direction.png & .pdf")
    plt.close()


def load_mvo_short_selling_data(results_dir='results/qp'):
    """Load MVO short selling constraint shift results."""
    csv_file = os.path.join(results_dir, 'constraint_shift_results.csv')
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)
    df['Method'] = df['method'].map(METHOD_MAP)
    df['Lower Bound'] = df['lower_bound']
    df['Regret'] = df['regret']
    return df


def plot_mvo_short_selling(df, save_dir='figures'):
    """Plot MVO short selling constraint shift results."""
    os.makedirs(save_dir, exist_ok=True)

    method_order = ['MSE', 'Projection', 'QPTH', 'CvxpyLayers', 'BPQP']
    method_order = [m for m in method_order if m in df['Method'].unique()]
    lower_bounds = sorted(df['Lower Bound'].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    markers = {'MSE': 'o', 'Projection': 'P', 'QPTH': 's', 'CvxpyLayers': '^', 'BPQP': 'D'}

    for method in method_order:
        method_data = df[df['Method'] == method]
        means = method_data.groupby('Lower Bound')['Regret'].mean()
        stds = method_data.groupby('Lower Bound')['Regret'].std()

        lw = 2.5 if method == 'Projection' else 1.5
        ms = 10 if method == 'Projection' else 7

        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=method, color=palette_mvo[method], marker=markers[method],
                    capsize=4, linewidth=lw, markersize=ms,
                    zorder=5 if method == 'Projection' else 3)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Train (Long-only)')
    ax.set_xlabel('Lower Bound (Short Selling Limit)')
    ax.set_ylabel('Regret')
    ax.set_title('MVO: Short Selling Generalization', fontweight='bold', pad=12)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'constraint_shift_mvo.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, 'constraint_shift_mvo.pdf'), facecolor='white')
    print(f"Saved {save_dir}/constraint_shift_mvo.png & .pdf")
    plt.close()


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_path)

    print("Loading constraint shift data...")

    df_cap = load_capacity_data('results/constraint_shift')
    df_dir = load_direction_data('results/constraint_shift')
    df_mvo = load_mvo_short_selling_data('results/qp')

    if df_cap is not None:
        print("\n=== Capacity Generalization ===")
        plot_capacity_generalization(df_cap)

    if df_dir is not None:
        print("\n=== Direction Generalization ===")
        plot_direction_generalization(df_dir)

    if df_mvo is not None:
        print("\n=== MVO Short Selling ===")
        plot_mvo_short_selling(df_mvo)

    print("\nDone!")
