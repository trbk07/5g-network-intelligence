"""Deterministic random seeds."""
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy; neural training seeds belong to its runtime."""
    random.seed(seed)
    np.random.seed(seed)


def set_neural_seed(seed: int) -> None:
    """Explicitly seed neural runtimes; preserve the lightweight legacy helper."""
    import torch

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
