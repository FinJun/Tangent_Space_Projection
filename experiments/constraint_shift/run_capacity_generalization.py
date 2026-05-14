#!/usr/bin/env python

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pkg"))

import pyepo
from pyepo.data import knapsack
from pyepo.data.dataset import optDataset

from projection.models import build_network, fcNet


def infer_architecture_from_state_dict(state_dict):
    arch = []
    i = 0
    while f'main.{i}.weight' in state_dict:
        weight = state_dict[f'main.{i}.weight']
        if not arch:
            arch.append(weight.shape[1])
        arch.append(weight.shape[0])
        i += 2
    return arch


def evaluate_capacity(net, weights, x_test, c_test, capacity_ratio, device):
    caps = capacity_ratio * weights.sum(axis=1)
    opt_model = pyepo.model.grb.knapsackModel(weights, caps)
    testset = optDataset(opt_model, x_test, c_test)
    testloader = DataLoader(testset, batch_size=32)
    regret = pyepo.metric.regret(net, opt_model, testloader)
    mse = pyepo.metric.MSE(net, testloader)
    return regret, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_item', type=int, default=100)
    parser.add_argument('--feat_dim', type=int, default=5)
    parser.add_argument('--deg', type=int, default=8)
    parser.add_argument('--model_dir', type=str, default='results/lp/models')
    parser.add_argument('--save_dir', type=str, default='results/constraint_shift')
    parser.add_argument('--capacity_ratios', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = f"{args.model_dir}/ks_{args.method}_d{args.deg}_seed{args.seed}.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    n_total = 1000 + 500 + 500
    weights, x, c = knapsack.genData(
        n_total, args.feat_dim, args.num_item,
        dim=5, deg=args.deg, noise_width=0, seed=args.seed
    )

    x_test, c_test = x[-500:], c[-500:]

    state_dict = torch.load(model_path, map_location=device)
    arch = infer_architecture_from_state_dict(state_dict)

    net = fcNet(arch)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    results = {
        'exp_type': 'capacity_generalization',
        'method': args.method,
        'deg': args.deg,
        'seed': args.seed,
        'num_item': args.num_item,
        'capacity_ratios': args.capacity_ratios,
        'regrets': {},
        'mses': {},
    }

    print(f"\n[{args.method}] Capacity Generalization Results:")
    print("-" * 50)

    for cap_ratio in args.capacity_ratios:
        regret, mse = evaluate_capacity(net, weights, x_test, c_test, cap_ratio, device)
        results['regrets'][str(cap_ratio)] = float(regret)
        results['mses'][str(cap_ratio)] = float(mse)

        marker = " (train)" if cap_ratio == 0.5 else ""
        print(f"  Capacity {cap_ratio:.1f}{marker}:  Regret={regret*100:.2f}%, MSE={mse:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f"{args.save_dir}/capacity_{args.method}_d{args.deg}_seed{args.seed}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
