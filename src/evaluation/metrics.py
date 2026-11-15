"""Finite-pair metrics with explicit exclusion counts and undefined-result flags."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """Metric value and observation coverage; undefined values use NaN plus a flag."""

    value: float
    n_valid: int
    n_excluded: int
    flags: tuple[str, ...] = ()


def _pairs(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and prediction shapes must match")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return actual[valid], predicted[valid], int(actual.size - valid.sum())


def _aggregate(actual: np.ndarray, predicted: np.ndarray, mode: str) -> MetricResult:
    a, p, excluded = _pairs(actual, predicted)
    if not a.size:
        return MetricResult(float("nan"), 0, excluded, ("no_valid_pairs",))
    if mode == "wape":
        denominator = np.abs(a).sum()
        if denominator == 0:
            return MetricResult(float("nan"), len(a), excluded, ("zero_denominator",))
        value = np.abs(a - p).sum() / denominator
    elif mode == "rmse":
        value = np.sqrt(np.mean((a - p) ** 2))
    else:
        value = np.abs(a - p).mean()
    return MetricResult(float(value), len(a), excluded)


def mae(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Return mean absolute error over finite pairs."""
    return _aggregate(actual, predicted, "mae")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Return root mean squared error over finite pairs."""
    return _aggregate(actual, predicted, "rmse")


def wape(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Return sum absolute error / sum absolute actual (ratio, not percent)."""
    return _aggregate(actual, predicted, "wape")


def _macro(actual: np.ndarray, predicted: np.ndarray, metric) -> MetricResult:
    a, p = np.asarray(actual), np.asarray(predicted)
    if a.ndim != 2 or a.shape != p.shape:
        raise ValueError("Macro metrics require matching (origin, beam) arrays")
    results = [metric(a[:, i], p[:, i]) for i in range(a.shape[1])]
    values = [r.value for r in results if np.isfinite(r.value)]
    excluded_beams = len(results) - len(values)
    flags = (f"excluded_beams={excluded_beams}",) if excluded_beams else ()
    if not values:
        flags += ("no_valid_beams",)
    return MetricResult(float(np.mean(values)) if values else float("nan"),
                        sum(r.n_valid for r in results), sum(r.n_excluded for r in results), flags)


def macro_wape(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Average defined per-beam WAPE, equally weighted; flag excluded beams."""
    return _macro(actual, predicted, wape)


def macro_mae(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Average defined per-beam MAE, equally weighted."""
    return _macro(actual, predicted, mae)


def active_only_mae(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Return MAE where actual > 0; excluded count includes inactive observations."""
    a, p, excluded = _pairs(actual, predicted)
    active = a > 0
    r = mae(a[active], p[active])
    return MetricResult(r.value, r.n_valid, excluded + int((~active).sum()), r.flags)


def zero_fp_rate(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0) -> MetricResult:
    """Fraction of actual-zero finite pairs predicted above epsilon (default strictly > 0)."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    a, p, excluded = _pairs(actual, predicted)
    zero = a == 0
    count = int(zero.sum())
    return MetricResult(float(np.mean(p[zero] > epsilon)) if count else float("nan"),
                        count, excluded + int((~zero).sum()),
                        () if count else ("no_zero_observations",))


def assign_load_tiers(fit_target: np.ndarray) -> np.ndarray:
    """Fit per-beam load tertiles on training-only (time, beam) target values.

    Equal means stay in the same tier. All-missing beams are Unknown; tied
    boundaries can leave a tier empty, rather than split identical loads arbitrarily.
    """
    values = np.asarray(fit_target, dtype=float)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("Expected nonempty (fit_time, beam) array")
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, values, 0).sum(0), counts,
                      out=np.full(values.shape[1], np.nan), where=counts > 0)
    tiers = np.full(values.shape[1], "Unknown", dtype="U7")
    observed = np.isfinite(means)
    if observed.any():
        cuts = np.quantile(means[observed], [1 / 3, 2 / 3])
        tiers[observed] = np.array(["Low", "Mid", "High"])[
            np.searchsorted(cuts, means[observed], side="left")]
    return tiers
