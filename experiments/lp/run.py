#!/usr/bin/env python

import os
import sys
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pkg"))

import pyepo
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


def build_opt_model(prob, grid=None, weights=None, caps=None):
    if prob == "sp":
        return pyepo.model.grb.shortestPathModel(grid)
    elif prob == "ks":
        return pyepo.model.grb.knapsackModel(weights, caps)
    else:
        raise ValueError(f"Unknown problem: {prob}")


def gen_data(prob, n_samples, feat_dim, deg, noise, seed, **kwargs):
    if prob == "sp":
        x, c = pyepo.data.shortestpath.genData(
            n_samples, feat_dim, kwargs["grid"],
            deg=deg, noise_width=noise, seed=seed
        )
        return x, c, None
    elif prob == "ks":
        weights, x, c = pyepo.data.knapsack.genData(
            n_samples, feat_dim, kwargs["num_item"],
            dim=5, deg=deg, noise_width=noise, seed=seed
        )
        caps = 0.5 * weights.sum(axis=1)
        return x, c, (weights, caps)
    else:
        raise ValueError(f"Unknown problem: {prob}")


class OptDatasetCached(Dataset):

    def __init__(self, feats, costs, sols, objs):
        self.feats = feats
        self.costs = costs
        self.sols = sols
        self.objs = objs

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )


def train(args):
    set_seed(args.seed)

    if args.method == "dbb" and args.prob == "ks" and args.smoothing == 20:
        args.smoothing = 10

    prob_kwargs = {}
    if args.prob == "sp":
        prob_kwargs["grid"] = tuple(args.grid)
        out_dim = (args.grid[0]-1)*args.grid[1] + args.grid[0]*(args.grid[1]-1)
    elif args.prob == "ks":
        prob_kwargs["num_item"] = args.num_item
        out_dim = args.num_item

    n_total = args.n_train + args.n_val + args.n_test
    cache_key_parts = [
        args.prob,
        f"seed{args.seed}",
        f"n{n_total}",
        f"feat{args.feat_dim}",
        f"deg{args.deg}",
        f"noise{args.noise}",
    ]
    if args.prob == "sp":
        cache_key_parts.append(f"grid{args.grid[0]}x{args.grid[1]}")
    elif args.prob == "ks":
        cache_key_parts.append(f"items{args.num_item}")
    cache_key = "_".join(cache_key_parts)

    use_cache = bool(args.data_cache_dir) and not args.no_data_cache
    cache_path = None
    cache_data = None
    if use_cache:
        os.makedirs(args.data_cache_dir, exist_ok=True)
        cache_path = os.path.join(args.data_cache_dir, f"{cache_key}.npz")

    if use_cache and cache_path and os.path.exists(cache_path):
        cache_data = np.load(cache_path, allow_pickle=True)
        x = cache_data["x"]
        c = cache_data["c"]
        if args.prob == "ks":
            weights = cache_data["weights"]
            caps = cache_data["caps"]
            ks_params = (weights, caps)
        else:
            ks_params = None
    else:
        x, c, ks_params = gen_data(args.prob, n_total, args.feat_dim, args.deg, args.noise, args.seed, **prob_kwargs)
        if use_cache and cache_path:
            if args.prob == "ks" and ks_params is not None:
                weights, caps = ks_params
                np.savez_compressed(cache_path, x=x, c=c, weights=weights, caps=caps)
            else:
                np.savez_compressed(cache_path, x=x, c=c)

    x_train, c_train = x[:args.n_train], c[:args.n_train]
    x_val, c_val = x[args.n_train:args.n_train+args.n_val], c[args.n_train:args.n_train+args.n_val]
    x_test, c_test = x[-args.n_test:], c[-args.n_test:]

    if args.prob == "ks" and ks_params is not None:
        prob_kwargs.pop("num_item", None)
        prob_kwargs["weights"], prob_kwargs["caps"] = ks_params
    opt_model = build_opt_model(args.prob, **prob_kwargs)

    train_sols = val_sols = test_sols = None
    train_objs = val_objs = test_objs = None
    if use_cache and cache_data is not None:
        if {"train_sols", "train_objs", "val_sols", "val_objs", "test_sols", "test_objs"}.issubset(cache_data.files):
            train_sols = cache_data["train_sols"]
            train_objs = cache_data["train_objs"]
            val_sols = cache_data["val_sols"]
            val_objs = cache_data["val_objs"]
            test_sols = cache_data["test_sols"]
            test_objs = cache_data["test_objs"]
        elif "sols" in cache_data and "objs" in cache_data:
            sols = cache_data["sols"]
            objs = cache_data["objs"]
            train_sols = sols[:args.n_train]
            train_objs = objs[:args.n_train]
            val_sols = sols[args.n_train:args.n_train + args.n_val]
            val_objs = objs[args.n_train:args.n_train + args.n_val]
            test_sols = sols[-args.n_test:]
            test_objs = objs[-args.n_test:]

    if (train_sols is None or val_sols is None or test_sols is None or
            train_objs is None or val_objs is None or test_objs is None):
        trainset_full = pyepo.data.dataset.optDataset(opt_model, x_train, c_train)
        valset_full = pyepo.data.dataset.optDataset(opt_model, x_val, c_val)
        testset_full = pyepo.data.dataset.optDataset(opt_model, x_test, c_test)

        train_sols, train_objs = trainset_full.sols, trainset_full.objs
        val_sols, val_objs = valset_full.sols, valset_full.objs
        test_sols, test_objs = testset_full.sols, testset_full.objs

        if use_cache and cache_path:
            if args.prob == "ks" and ks_params is not None:
                weights, caps = ks_params
                np.savez_compressed(
                    cache_path,
                    x=x, c=c,
                    train_sols=train_sols, train_objs=train_objs,
                    val_sols=val_sols, val_objs=val_objs,
                    test_sols=test_sols, test_objs=test_objs,
                    weights=weights, caps=caps
                )
            else:
                np.savez_compressed(
                    cache_path,
                    x=x, c=c,
                    train_sols=train_sols, train_objs=train_objs,
                    val_sols=val_sols, val_objs=val_objs,
                    test_sols=test_sols, test_objs=test_objs
                )

    trainset = OptDatasetCached(x_train, c_train, train_sols, train_objs)
    valset = OptDatasetCached(x_val, c_val, val_sols, val_objs)
    testset = OptDatasetCached(x_test, c_test, test_sols, test_objs)

    shuffle_train = (args.method != "lava")
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=shuffle_train)
    valloader = DataLoader(valset, batch_size=args.batch)
    testloader = DataLoader(testset, batch_size=args.batch)

    net = build_network(args.feat_dim, out_dim, args.hidden if args.hidden else None)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    trainer_kwargs = {}
    if args.method == "spo":
        trainer_kwargs["epsilon"] = args.epsilon
    elif args.method == "dbb":
        trainer_kwargs["smoothing"] = args.smoothing
        trainer_kwargs["epsilon"] = args.epsilon
    elif args.method == "pfyl":
        trainer_kwargs["n_samples"] = args.n_samples
        trainer_kwargs["sigma"] = args.sigma
        trainer_kwargs["epsilon"] = args.epsilon
    elif args.method == "projection":
        trainer_kwargs["epsilon"] = args.epsilon
        trainer_kwargs["forward_smoothing"] = args.fwd_smooth
        trainer_kwargs["ni_eta"] = args.ni_eta
        trainer_kwargs["ni_reg"] = args.ni_reg
        trainer_kwargs["ni_delta"] = args.ni_delta
    elif args.method == "lava":
        trainer_kwargs["threshold"] = args.threshold

    noise_str = f"_noise{args.noise}" if args.noise > 0 else ""
    save_path = f"{args.save_dir}/models/{args.prob}_{args.method}_d{args.deg}{noise_str}_seed{args.seed}.pt"

    trainer = create_trainer(
        args.method, net, opt_model, optimizer,
        epochs=args.epochs, save_path=save_path, **trainer_kwargs
    )

    t0 = time.time()

    if args.method == "lava":
        from projection.training.lava import optDatasetAugmented, collate_fn_lava, shortestPathModelLAVA, knapsackModelLAVA

        print("Creating augmented dataset with adjacent vertices...")

        if args.prob == "sp":
            lava_opt_model = shortestPathModelLAVA(tuple(args.grid))
        elif args.prob == "ks":
            weights, caps = ks_params
            lava_opt_model = knapsackModelLAVA(weights, caps)
        else:
            raise NotImplementedError(f"LAVA not implemented for {args.prob}")

        trainset_aug = optDatasetAugmented(lava_opt_model, x_train, c_train)
        trainloader = DataLoader(trainset_aug, batch_size=args.batch, shuffle=False, collate_fn=collate_fn_lava)

        adj_verts_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.FloatTensor(av) for av in trainset_aug.adjacent_verts],
            batch_first=True, padding_value=0
        )
        trainer.set_adjacent_vertices(adj_verts_tensor)

    net = trainer.train(trainloader, valloader)
    elapsed = time.time() - t0

    regret = pyepo.metric.regret(net, opt_model, testloader)
    mse = pyepo.metric.MSE(net, testloader)

    print(f"\n[{args.method}] Regret: {regret*100:.2f}%, MSE: {mse:.4f}, Time: {elapsed:.1f}s")

    return {"method": args.method, "deg": args.deg, "noise": args.noise, "regret": regret, "mse": mse, "time": elapsed, "seed": args.seed}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prob", type=str, default="sp", choices=["sp", "ks"])
    parser.add_argument("--grid", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--num_item", type=int, default=100)

    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--feat_dim", type=int, default=5)
    parser.add_argument("--deg", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.0)

    parser.add_argument("--method", type=str, default="spo",
                        choices=["mse", "spo", "dbb", "pfyl", "projection", "lava"])
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, nargs="*", default=[])

    parser.add_argument("--smoothing", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--fwd_smooth", type=float, default=0.1)
    parser.add_argument("--ni_eta", type=float, default=0.1)
    parser.add_argument("--ni_reg", type=float, default=1e-8)
    parser.add_argument("--ni_delta", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=-0.1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results/lp")
    parser.add_argument("--data_cache_dir", type=str, default="results/lp_data_cache")
    parser.add_argument("--no_data_cache", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    results = []
    base_seed = args.seed
    for run in range(args.n_runs):
        args.seed = base_seed + run
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{args.n_runs}: {args.prob} / {args.method} (seed={args.seed})")
        print(f"{'='*60}")
        result = train(args)
        results.append(result)

    df = pd.DataFrame(results)

    noise_str = f"_noise{args.noise}" if args.noise > 0 else ""
    n_train_str = f"_n{args.n_train}" if args.n_train != 1000 else ""

    if args.n_runs == 1:
        fname = f"{args.save_dir}/{args.prob}_{args.method}_d{args.deg}{noise_str}{n_train_str}_seed{base_seed}.csv"
    else:
        fname = f"{args.save_dir}/{args.prob}_{args.method}_d{args.deg}{noise_str}{n_train_str}.csv"
    df.to_csv(fname, index=False)
    print(f"\nSaved to {fname}")

    print(f"\n[Summary] Regret: {df['regret'].mean()*100:.2f}% ± {df['regret'].std()*100:.2f}%")


if __name__ == "__main__":
    main()
