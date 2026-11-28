"""Explicit horizon-balanced training objective for partial-label research."""

import torch


def horizon_balanced_l1(predicted: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Average per-horizon observed MAE, then average horizons with supervision.

    Missing labels never become numerical targets. Use the same objective in
    the joint-label control and partial-label candidate. Stopping remains the
    historical joint-window criterion; this function changes training only.
    """
    if predicted.shape != actual.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("Aligned batch/cell/three-horizon predictions and targets required")
    if not torch.isfinite(predicted).all() or torch.isinf(actual).any():
        raise ValueError("Finite predictions and finite-or-missing targets required")
    losses, count = [], 0
    for horizon in range(3):
        mask = torch.isfinite(actual[..., horizon])
        valid = int(mask.sum())
        if valid:
            losses.append(torch.abs(predicted[..., horizon][mask]-actual[..., horizon][mask]).mean())
            count += valid
    return (torch.stack(losses).mean() if losses else predicted.sum()*0), count
