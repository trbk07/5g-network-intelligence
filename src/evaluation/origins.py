"""Integer forecast origins; all split endpoints are inclusive."""
import numpy as np


def _validate(horizon: int, lookback: int, stride: int = 1) -> None:
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x < 1
           for x in (horizon, lookback, stride)):
        raise ValueError("horizon, lookback and stride must be positive integers")


def train_origins(fit_end: int, horizon: int, lookback: int = 168,
                  stride: int = 1) -> np.ndarray:
    """Return origins whose full history and target lie in the fit segment."""
    _validate(horizon, lookback, stride)
    return np.arange(lookback - 1, fit_end - horizon + 1, stride, dtype=np.int64)


def evaluation_origins(fit_end: int, segment_end: int, horizon: int,
                       lookback: int = 168) -> np.ndarray:
    """Use fixed-model rolling origins from fit_end, including observed evaluation history."""
    _validate(horizon, lookback)
    if segment_end <= fit_end:
        raise ValueError("Evaluation segment must follow fitted data")
    return np.arange(max(lookback - 1, fit_end), segment_end - horizon + 1,
                     dtype=np.int64)


def train_origins_joint(fit_end: int, horizons: tuple[int, ...] = (1, 6, 24),
                        lookback: int = 168, stride: int = 1) -> np.ndarray:
    """Return training origins that have all requested horizon targets."""
    if not horizons:
        raise ValueError("At least one horizon is required")
    for horizon in horizons:
        _validate(horizon, lookback, stride)
    return train_origins(fit_end, max(horizons), lookback, stride)


def assert_no_target_overlap(segments: dict[str, dict[int, np.ndarray]]) -> None:
    """Reject target overlap between segments, considering every horizon together."""
    seen: set[int] = set()
    for name, horizon_origins in segments.items():
        targets = {int(origin + h) for h, origins in horizon_origins.items() for origin in origins}
        if seen & targets:
            raise ValueError(f"Target overlap at segment {name}: {sorted(seen & targets)[:5]}")
        seen |= targets


def split_origins(split: dict) -> dict[str, dict[int, np.ndarray]]:
    """Build train/validation/holdout origin sets without reading any observations."""
    lookback = split["lookback"]
    result = {
        "train": {h: train_origins(split["fit"][1], h, lookback) for h in split["horizons"]},
        "validation": {h: evaluation_origins(split["fit"][1], split["validation"][1], h,
                                              lookback) for h in split["horizons"]},
        "holdout": {h: evaluation_origins(split["final_refit"][1], split["holdout"][1], h,
                                           lookback) for h in split["horizons"]},
    }
    assert_no_target_overlap(result)
    return result
