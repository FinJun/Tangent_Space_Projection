"""Default configurations for DFL experiments.

These settings match the original Tangent_space_projection codebase.
"""

# =============================================================================
# Training defaults
# =============================================================================
TRAINING_DEFAULTS = {
    "epochs": 10000,
    "batch": 32,
    "lr": 1e-2,
    "patience": 3,           # Early stopping patience (checked every 10 epochs)
    "min_improvement": 0.01, # Minimum relative improvement for early stopping
    "max_time": 600,         # Max training time in seconds (10 min)
}

# =============================================================================
# Method-specific defaults
# =============================================================================
METHOD_DEFAULTS = {
    "mse": {},
    "spo": {},
    "dbb": {"smoothing": 20},
    "dpo": {"n_samples": 1, "sigma": 1.0},
    "pfyl": {"n_samples": 1, "sigma": 1.0},
    "projection": {"epsilon": 0.01, "forward_smoothing": 0.1},
}

# =============================================================================
# Problem defaults
# =============================================================================
PROBLEM_DEFAULTS = {
    "sp": {  # Shortest Path
        "grid": (5, 5),
        "feat_dim": 5,
        "deg": 8,
    },
    "ks": {  # Knapsack
        "num_item": 150,
        "dim": 5,
        "feat_dim": 5,
        "deg": 8,
    },
}

# =============================================================================
# Data defaults
# =============================================================================
DATA_DEFAULTS = {
    "n_train": 1000,
    "n_val": 500,
    "n_test": 500,
    "noise": 0.0,
}

# =============================================================================
# Experiment seeds
# =============================================================================
SEEDS = [0, 1, 2, 3, 4]

# =============================================================================
# Noise levels for robustness experiments
# =============================================================================
NOISE_LEVELS = [0.0, 0.25, 0.5]
