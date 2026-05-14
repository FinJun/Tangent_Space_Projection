from .base import BaseTrainer, EarlyStopping
from .trainers import (
    MSETrainer,
    SPOTrainer,
    DBBTrainer,
    PFYLTrainer,
    NormalInjectionProjectionTrainer,
    LAVATrainer,
)
from .lava import (
    LAVALoss,
    precompute_adjacent_vertices,
    pad_adjacent_vertices,
    optDatasetAugmented,
    collate_fn_lava,
)

TRAINERS = {
    "mse": MSETrainer,
    "spo": SPOTrainer,
    "dbb": DBBTrainer,
    "pfyl": PFYLTrainer,
    "projection": NormalInjectionProjectionTrainer,
    "lava": LAVATrainer,
}


def create_trainer(method, net, opt_model, optimizer, **kwargs):
    if method not in TRAINERS:
        raise ValueError(f"Unknown method: {method}. Available: {list(TRAINERS.keys())}")
    return TRAINERS[method](net, opt_model, optimizer, **kwargs)
