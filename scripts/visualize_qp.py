#!/usr/bin/env python
"""Visualize QP/MVO experiment results."""

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
    'BPQP': '#5B7C99',
    'CVXPY': '#7A9A7A',
    'QPTH': '#8B7B96',
    'Projection': '#B8433E',
}

METHOD_MAP = {
    'mse': 'MSE',
    'bpqp': 'BPQP',
    'cvxpy': 'CVXPY',
    'qpth': 'QPTH',
    'projection': 'Projection',
    'projection_eps': 'Projection',
}


def load_qp_data(results_dir='results/qp', risk_aversion=1.0):
    """Load QP/MVO experiment results from JSON files."""
    data = []
    pattern = f'mvo_*_lambda{risk_aversion}_S*.json'
    for json_file in glob.glob(os.path.join(results_dir, pattern)):
        with open(json_file, 'r') as f:
            result = json.load(f)
            if result['method'] not in METHOD_MAP:
                continue
            method = METHOD_MAP[result['method']]
            data.append({
                'Method': method,
                'Regret (%)': result['test_regret'] * 100,
                'MSE': result['test_mse'],
                'Time (s)': result['train_time'],
                'seed': result['seed']
            })

    if not data:
        print(f"No MVO result files found in {results_dir}")
        return None

    return pd.DataFrame(data)


def plot_qp_results(df, save_dir='figures'):
    """Plot QP/MVO experiment results."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    method_order = ['MSE', 'BPQP', 'CVXPY', 'QPTH', 'Projection']
    method_order = [m for m in method_order if m in df['Method'].unique()]

    # Regret Plot
    sns.barplot(data=df, x='Method', y='Regret (%)', hue='Method',
                palette=palette, ax=axes[0], capsize=.08, dodge=False,
                order=method_order, errorbar='sd',
                err_kws={'linewidth': 1.5}, edgecolor='white', linewidth=1.5)
    axes[0].set_title("(a) Decision Quality", fontweight='bold', pad=10)
    axes[0].set_ylabel("Regret (%)")
    axes[0].set_xlabel("")
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
    if axes[1].get_legend():
        axes[1].get_legend().remove()

    plt.tight_layout(w_pad=3)
    plt.savefig(os.path.join(save_dir, 'qp_mvo_results.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, 'qp_mvo_results.pdf'), facecolor='white')
    print(f"Saved {save_dir}/qp_mvo_results.png & .pdf")
    plt.close()


def plot_regret_vs_time(df, save_dir='figures'):
    """Plot Regret vs Training Time scatter plot."""
    os.makedirs(save_dir, exist_ok=True)

    method_order = ['MSE', 'BPQP', 'CVXPY', 'QPTH', 'Projection']
    method_order = [m for m in method_order if m in df['Method'].unique()]

    data_points = {}
    for method in method_order:
        method_data = df[df['Method'] == method]
        if len(method_data) > 0:
            regret_mean = method_data['Regret (%)'].mean()
            time_mean = method_data['Time (s)'].mean()
            data_points[method] = (time_mean, regret_mean)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in data_points:
        time_mean, regret_mean = data_points[method]
        color = palette[method]
        ax.scatter(time_mean, regret_mean, s=150, color=color, zorder=5,
                   edgecolor='white', linewidth=1.5, label=method)
        ax.annotate(method, (time_mean, regret_mean),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=11, va='center', ha='left')

    times = [t for t, _ in data_points.values()]
    ax.set_xlim(0, max(times) * 1.15)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Regret (%)")
    ax.set_title("MVO: Decision Quality vs Training Efficiency", fontweight='bold', pad=12)

    plt.savefig(os.path.join(save_dir, 'qp_regret_vs_time.png'), dpi=300, facecolor='white')
    plt.savefig(os.path.join(save_dir, 'qp_regret_vs_time.pdf'), facecolor='white')
    print(f"Saved {save_dir}/qp_regret_vs_time.png & .pdf")
    plt.close()


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_path)

    print("Loading QP/MVO data...")

    df_qp = load_qp_data('results/qp', risk_aversion=1.0)

    if df_qp is not None:
        print("\n=== MVO Results ===")
        summary = df_qp.groupby('Method').agg({
            'Regret (%)': ['mean', 'std'],
            'MSE': ['mean', 'std'],
            'Time (s)': ['mean', 'std'],
        }).round(4)
        print(summary)

        plot_qp_results(df_qp)
        plot_regret_vs_time(df_qp)

    print("\nDone!")
