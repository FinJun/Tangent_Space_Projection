#!/usr/bin/env python

import os
import sys
import json
import time
import random
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pkg"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import pyepo
from pyepo.data import shortestpath
from pyepo.data.dataset import optDataset

from projection.training import create_trainer
from projection.models import build_network


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_on_model(net, opt_model, x_test, c_test, device):
    net.eval()
    testset = optDataset(opt_model, x_test, c_test)
    testloader = DataLoader(testset, batch_size=32)
    regret = pyepo.metric.regret(net, opt_model, testloader)
    mse = pyepo.metric.MSE(net, testloader)
    return regret, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deg', type=int, default=8)
    parser.add_argument('--grid', type=int, nargs=2, default=[5, 5])
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='results/constraint_shift')
    parser.add_argument('--model_dir', type=str, default='results/lp/models')
    parser.add_argument('--shift', type=str, default='cross',
                        choices=['reverse', 'cross'])
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--fwd_smooth', type=float, default=0.1)
    parser.add_argument('--ni_eta', type=float, default=0.1)
    parser.add_argument('--ni_reg', type=float, default=1e-8)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    grid = tuple(args.grid)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_total = args.n_train + 500 + 500
    x, c = shortestpath.genData(n_total, 5, grid, deg=args.deg, noise_width=0, seed=args.seed)

    x_train, c_train = x[:args.n_train], c[:args.n_train]
    x_val, c_val = x[args.n_train:args.n_train+500], c[args.n_train:args.n_train+500]
    x_test, c_test = x[-500:], c[-500:]

    opt_model_fwd = pyepo.model.grb.shortestPathModel(grid, reverse=False)

    if args.shift == "reverse":
        opt_model_shift = pyepo.model.grb.shortestPathModel(grid, reverse=True)
    else:
        opt_model_shift = pyepo.model.grb.shortestPathModel(grid, cross=True)

    net = build_network(5, opt_model_fwd.num_cost)

    model_path = args.load_model or os.path.join(
        args.model_dir, f"sp_{args.method}_d{args.deg}_seed{args.seed}.pt"
    )

    if not args.train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Run LP training or pass --train to train here."
            )
        print(f"Loading pre-trained model from {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        elapsed = 0.0
    else:
        print(f"Training on FORWARD direction...")
        trainset = optDataset(opt_model_fwd, x_train, c_train)
        valset = optDataset(opt_model_fwd, x_val, c_val)

        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        valloader = DataLoader(valset, batch_size=32)

        lr = 1e-2
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        trainer_kwargs = {}
        if args.method == "dbb":
            trainer_kwargs["smoothing"] = 20
        elif args.method == "pfyl":
            trainer_kwargs["n_samples"] = 1
            trainer_kwargs["sigma"] = 1.0
        elif args.method == "projection":
            trainer_kwargs["epsilon"] = args.epsilon
            trainer_kwargs["forward_smoothing"] = args.fwd_smooth
            trainer_kwargs["ni_eta"] = args.ni_eta
            trainer_kwargs["ni_reg"] = args.ni_reg

        os.makedirs(f"{args.save_dir}/models", exist_ok=True)
        save_path = f"{args.save_dir}/models/direction_{args.method}_d{args.deg}_seed{args.seed}.pt"

        trainer = create_trainer(
            args.method, net, opt_model_fwd, optimizer,
            epochs=args.epochs, save_path=save_path, **trainer_kwargs
        )

        t0 = time.time()
        net = trainer.train(trainloader, valloader)
        elapsed = time.time() - t0

    net.to(device)
    net.eval()

    regret_fwd, mse_fwd = evaluate_on_model(net, opt_model_fwd, x_test, c_test, device)
    regret_shift, mse_shift = evaluate_on_model(net, opt_model_shift, x_test, c_test, device)

    result = {
        'exp_type': 'direction_generalization',
        'method': args.method,
        'deg': args.deg,
        'seed': args.seed,
        'grid': list(grid),
        'regret_forward': float(regret_fwd),
        'shift': args.shift,
        'regret_shift': float(regret_shift),
        'mse_forward': float(mse_fwd),
        'mse_shift': float(mse_shift),
        'time': float(elapsed),
    }

    if args.shift == "reverse":
        result['regret_reverse'] = float(regret_shift)
        result['mse_reverse'] = float(mse_shift)
    else:
        result['regret_cross'] = float(regret_shift)
        result['mse_cross'] = float(mse_shift)

    print(f"\n[{args.method}] Direction Generalization Results:")
    print(f"  Forward (train):  Regret={regret_fwd*100:.2f}%, MSE={mse_fwd:.4f}")
    print(f"  {args.shift.title()} (test):   Regret={regret_shift*100:.2f}%, MSE={mse_shift:.4f}")
    print(f"  Regret increase:  {(regret_shift - regret_fwd)*100:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/direction_{args.method}_d{args.deg}_seed{args.seed}.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    main()
