"""Paired cluster bootstrap; resample whole verified BS clusters, preserving paired errors."""

import numpy as np


def _stats(actual, a, b, clusters, mask):
    if actual.shape != a.shape or actual.shape != b.shape or mask.shape != actual.shape:
        raise ValueError("All paired arrays must have matching shapes")
    labels = np.unique(clusters)
    if len(clusters) != actual.shape[1]:
        raise ValueError("One cluster label per beam is required")
    if np.any(mask & ~(np.isfinite(actual) & np.isfinite(a) & np.isfinite(b))):
        raise ValueError("Bootstrap mask includes a nonfinite comparison pair")
    sums, counts = [], []
    delta = np.abs(a - actual) - np.abs(b - actual)
    for label in labels:
        selected = (clusters == label)[None, :] & mask
        sums.append(float(np.where(selected, delta, 0).sum()))
        counts.append(int(selected.sum()))
    return np.asarray(sums), np.asarray(counts)


def paired_multihorizon_ci(
    pairs: list[tuple],
    clusters: np.ndarray,
    scales: list[float],
    repetitions: int = 1000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict:
    """Mean normalized horizon MAE difference; use identical BS draws across horizons."""
    if not pairs or len(pairs) != len(scales) or repetitions < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid paired bootstrap configuration")
    if any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Normalization scales must be positive")
    clusters = np.asarray(clusters)
    summaries = [_stats(pair[0], pair[1], pair[2], clusters, pair[3]) for pair in pairs]
    sums = np.stack([s / scales[i] for i, (s, _) in enumerate(summaries)])
    counts = np.stack([c for _, c in summaries])
    used = counts.sum(0) > 0
    sums, counts = sums[:, used], counts[:, used]
    if not counts.size or np.any(counts.sum(1) == 0) or used.sum() < 2:
        return {
            "difference": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "status": "insufficient_clusters",
            "n_clusters": int(used.sum()),
        }
    n = int(used.sum())
    draws = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), repetitions)
    denominator = counts @ draws.T
    valid = (denominator > 0).all(0)
    bootstrap = ((sums @ draws[valid].T) / denominator[:, valid]).mean(0)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(bootstrap, [alpha, 1 - alpha])
    return {
        "difference": float((sums.sum(1) / counts.sum(1)).mean()),
        "lower": float(low),
        "upper": float(high),
        "status": "ok",
        "n_clusters": n,
        "repetitions": repetitions,
        "seed": seed,
    }


def paired_cluster_ci(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray | None = None,
    repetitions: int = 1000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict:
    """CI for MAE(A)-MAE(B); negative values favor A, using a shared finite-pair mask."""
    if mask is None:
        from src.evaluation.common_mask import comparable_mask

        mask = comparable_mask(y_true, {"a": prediction_a, "b": prediction_b})
    return paired_multihorizon_ci(
        [(y_true, prediction_a, prediction_b, mask)], clusters, [1.0], repetitions, seed, confidence
    )
