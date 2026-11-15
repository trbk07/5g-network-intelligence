"""Deterministic random seeds."""
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy; neural training seeds belong to its runtime."""
    random.seed(seed)
    np.random.seed(seed)
