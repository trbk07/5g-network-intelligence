"""Pure inference for a zero-mass plus positive-quantile mixture."""

import numpy as np


def mixture_median(active_probability, positive_quantiles, levels):
    """Approximate mixture median; no probability calibration or learned thresholds.

    For P(Y>0)=p>0.5, the positive conditional quantile is 1-0.5/p.
    Otherwise zero is a median. Interpolation includes the fixed nonnegative
    lower boundary (0,0); sparse quantiles and crossing repair are approximations.
    """
    p = np.asarray(active_probability, dtype=float)
    values = np.asarray(positive_quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if (p.ndim != 1 or values.shape != (len(p), len(levels))
            or levels.ndim != 1 or not len(levels)
            or not np.isfinite(levels).all() or levels[0] <= 0
            or levels[-1] != 0.5 or not (np.diff(levels) > 0).all()):
        raise ValueError("Expected ordered conditional levels ending at 0.5")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(values).all():
        raise ValueError("Finite probabilities in [0,1] and conditional predictions required")
    q = np.zeros_like(p)
    active = p > 0.5
    q[active] = 1 - 0.5 / p[active]
    grid = np.r_[0, levels]
    repaired = np.column_stack([np.zeros(len(p)), np.sort(np.maximum(values, 0), axis=1)])
    upper = np.searchsorted(grid, q, side="right").clip(1, len(grid) - 1)
    lower = upper - 1
    fraction = (q - grid[lower]) / (grid[upper] - grid[lower])
    index = np.arange(len(p))
    result = repaired[index, lower] * (1 - fraction) + repaired[index, upper] * fraction
    return np.where(active, result, 0)
