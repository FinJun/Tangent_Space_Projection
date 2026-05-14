"""Optimization model builders."""

import pyepo


def build_model(problem, grid=None, weights=None, caps=None):
    """Build PyEPO optimization model."""
    if problem == "sp":
        return pyepo.model.grb.shortestPathModel(grid)
    elif problem == "ks":
        return pyepo.model.grb.knapsackModel(weights, caps)
    else:
        raise ValueError(f"Unknown problem: {problem}")


def get_output_dim(problem, grid=None, num_item=None):
    """Get output dimension for problem."""
    if problem == "sp":
        h, w = grid
        return (h - 1) * w + (w - 1) * h
    elif problem == "ks":
        return num_item
    else:
        raise ValueError(f"Unknown problem: {problem}")
