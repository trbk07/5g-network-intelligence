"""Exact replay of hosted float32 uniform initialization on Windows CPUs."""

from unittest.mock import patch

import numpy as np
import torch


def verify_cold_state(factory, seed, observed):
    """Require bitwise equality to native or hosted-FMA seeded construction."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = factory()
    for policy, model in (("native", native), ("hosted_fma", hosted_cold_model(factory, seed))):
        expected = model.state_dict()
        if set(expected) == set(observed) and all(torch.equal(v, observed[k]) for k, v in expected.items()):
            return policy
    raise ValueError("Cold weights do not exactly reproduce the declared seed under native or hosted FMA arithmetic")


def hosted_cold_model(factory, seed):
    """Recreate Linux FMA uniform arithmetic without changing RNG or fitting.

    Windows uniform kernels round multiply and add separately. Hosted Linux
    uses one FMA rounding. Float64 represents the float32 product and sum
    accurately enough to reproduce the final float32 FMA result exactly.
    Scope this patch to a cold model constructor; never use it during training.
    """
    original = torch.Tensor.uniform_

    def fused_uniform(tensor, a=0., b=1., *, generator=None):
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError("Hosted initialization replay supports CPU float32 only")
        original(tensor, 0., 1., generator=generator)
        low, high = np.float32(a), np.float32(b)
        width = np.float32(high-low)
        tensor.copy_((tensor.double()*float(width)+float(low)).float())
        return tensor

    with torch.random.fork_rng(devices=[]), patch.object(torch.Tensor, "uniform_", fused_uniform):
        torch.manual_seed(seed)
        return factory()
