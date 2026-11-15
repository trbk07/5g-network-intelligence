"""Reproducible stratified beam sampling from training-only statistics."""
import numpy as np

from src.features.static_stats import StaticStats


def sample_beams(static: StaticStats, size: int = 600, seed: int = 17) -> np.ndarray:
    """Sample proportionally across joint tertiles of mean/std/active ratio."""
    if not isinstance(size, int) or size < 1:
        raise ValueError("sample size must be positive")
    valid = np.flatnonzero(np.isfinite(static.values).all(1))
    if not valid.size:
        raise ValueError("No beams have observed fit targets")
    size = min(size, len(valid))
    features = static.values[valid][:, [0, 1, 3]]
    bins = np.stack([np.searchsorted(np.quantile(features[:, k], [1/3, 2/3]),
                                     features[:, k], side="left") for k in range(3)], axis=1)
    strata = bins @ np.array([9, 3, 1])
    labels, counts = np.unique(strata, return_counts=True)
    desired = counts * size / counts.sum()
    allocations = np.floor(desired).astype(int)
    for i in np.argsort(-(desired - allocations), kind="stable")[:size - allocations.sum()]:
        allocations[i] += 1
    rng = np.random.default_rng(seed)
    selected = [rng.choice(valid[strata == label], count, replace=False)
                for label, count in zip(labels, allocations) if count]
    return np.sort(np.concatenate(selected))
