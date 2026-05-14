"""Data generation utilities."""

import pyepo


def generate_data(problem, n_samples, feat_dim, deg=1, noise=0.0, seed=0, **kwargs):
    """Generate synthetic data for optimization problems."""
    if problem == "sp":
        return pyepo.data.shortestpath.genData(
            n_samples, feat_dim, kwargs["grid"],
            deg=deg, noise_width=noise, seed=seed
        )
    elif problem == "ks":
        return pyepo.data.knapsack.genData(
            n_samples, feat_dim, kwargs["num_item"],
            dim=2, deg=deg, noise_width=noise, seed=seed
        )
    else:
        raise ValueError(f"Unknown problem: {problem}")
