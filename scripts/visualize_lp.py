#!/usr/bin/env python
"""Visualize LP experiment results (Shortest Path, Knapsack)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
}

METHOD_MAP = {
    'mse': 'MSE',
    'spo': 'SPO+',
    'dbb': 'DBB',
    'pfyl': 'PFYL',
    'lava': 'LAVA',
    'projection': 'Projection'
}


def load_lp_data(results_dir='results/lp', problem='sp'):
    """Load LP experiment results from CSV files."""
    data = []
    for csv_file in glob.glob(os.path.join(results_dir, f'{problem}_*.csv')):
        df = pd.read_csv(csv_file)
        data.append(df)

    if not data:
        print(f"No CSV files found in {results_dir} for problem={problem}")
        return None

    df = pd.concat(data, ignore_index=True)
    df['Method'] = df['method'].map(METHOD_MAP)
    df['Regret (%)'] = df['regret'] * 100
    df['MSE'] = df['mse']
    return df


def plot_lp_results(df, save_dir='figures', problem='sp'):
    """Plot LP experiment results (Regret & MSE)."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    method_order = ['MSE', 'SPO+', 'DBB', 'PFYL', 'LAVA', 'Projection']
    problem_name = 'Shortest Path' if problem == 'sp' else 'Knapsack'

    # Regret Plot
    sns.barplot(data=df, x='Method', y='Regret (%)', hue='Method',
                palette=palette, ax=axes[0], capsize=.08, dodge=False,
                order=method_order, errorbar='sd',
                err_kws={'linewidth': 1.5}, edgecolor='white', linewidth=1.5)
    axes[0].set_title("(a) Decision Quality", fontweight='bold', pad=10)
    axes[0].set_ylabel("Regret (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylim(bottom=0)
    if axes[0].get_legend():
        axes[0].get_legend().remove()

    # MSE Plot
    sns.barplot(data=df, x='Method', y='MSE', hue='Method',
                palette=palette, ax=axes[1], capsize=.08, dodge=False,
                order=method_order, errorbar='sd',
                err_kws={'linewidth': 1.5}, edgecolor='white', linewidth=1.5)
    axes[1].set_title("(b) Prediction Accuracy", fontweight='bold', pad=10)
    axes[1].set_ylabel("MSE")
    axes[1].set_xlabel("")
    axes[1].set_ylim(bottom=0)
    if axes[1].get_legend():
        axes[1].get_legend().remove()

    plt.tight_layout(w_pad=3)
    filename = f'{problem}_lp_results'
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'), facecolor='white')
    print(f"Saved {save_dir}/{filename}.png & .pdf")
    plt.close()


def plot_regret_vs_time(df, save_dir='figures', problem='sp'):
    """Plot Regret vs Training Time scatter plot."""
    os.makedirs(save_dir, exist_ok=True)

    method_order = ['MSE', 'SPO+', 'DBB', 'PFYL', 'LAVA', 'Projection']
    problem_name = 'Shortest Path' if problem == 'sp' else 'Knapsack'

    data_points = {}
    for method in method_order:
        method_data = df[df['Method'] == method]
        if len(method_data) > 0:
            regret_mean = method_data['Regret (%)'].mean()
            time_mean = method_data['time'].mean()
            data_points[method] = (time_mean, regret_mean)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in data_points:
        time_mean, regret_mean = data_points[method]
        color = palette[method]
        ax.scatter(time_mean, regret_mean, s=150, color=color, zorder=5,
                   edgecolor='white', linewidth=1.5)
        ax.annotate(method, (time_mean, regret_mean),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=11, va='center', ha='left')

    times = [t for t, _ in data_points.values()]
    ax.set_xlim(0, max(times) * 1.15)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Regret (%)")
    ax.set_title(f"{problem_name}: Decision Quality vs Training Efficiency", fontweight='bold', pad=12)

    filename = f'{problem}_regret_vs_time'
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'), facecolor='white')
    print(f"Saved {save_dir}/{filename}.png & .pdf")
    plt.close()


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_path)

    print("Loading LP data...")

    df_sp = load_lp_data('results/lp', problem='sp')
    df_ks = load_lp_data('results/lp', problem='ks')

    if df_sp is not None:
        print("\n=== Shortest Path ===")
        plot_lp_results(df_sp, problem='sp')
        plot_regret_vs_time(df_sp, problem='sp')

    if df_ks is not None:
        print("\n=== Knapsack ===")
        plot_lp_results(df_ks, problem='ks')
        plot_regret_vs_time(df_ks, problem='ks')

    print("\nDone!")
