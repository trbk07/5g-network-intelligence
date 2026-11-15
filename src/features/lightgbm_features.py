"""Vectorized 44-feature design; rows are ordered origin-major then beam-major."""
import numpy as np

from src.data.schema import KPIS
from src.features.static_stats import StaticStats

LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 167)
FEATURE_NAMES = (
    [f"DLPRB_lag_{lag}" for lag in LAGS]
    + [f"DLPRB_target_seasonal_{day}d" for day in (1, 2, 3, 7)]
    + ["DLPRB_target_seasonal_mean_7d"]
    + [f"DLPRB_roll_{window}_{stat}" for window in (6, 24, 168)
       for stat in ("mean", "std", "max")]
    + ["target_phase_24", "target_phase_168"]
    + [f"{kpi}_{suffix}" for kpi in KPIS[1:]
       for suffix in ("lag_0", "lag_1", "lag_24", "roll_6_mean", "roll_24_mean")]
    + ["beam_fit_mean", "beam_fit_std", "beam_fit_p90", "beam_fit_active_ratio"]
)


def build_features(values: np.ndarray, origins: np.ndarray, horizon: int,
                   static: StaticStats, phase_origins: np.ndarray | None = None) -> np.ndarray:
    """Build float32 features from observed indices only, without sliding-window tensors.

    phase_origins supplies absolute origins when values is a local history window.
    Rolling std is population std (ddof=0); missing-only windows remain NaN.
    """
    values, origins = np.asarray(values), np.asarray(origins)
    if (values.ndim != 3 or values.shape[2] != 4 or origins.ndim != 1 or not origins.size
            or not np.issubdtype(origins.dtype, np.integer)):
        raise ValueError("Expected (time, beam, 4) values and nonempty integer origins")
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 24:
        raise ValueError("Feature horizons must lie in 1..24")
    if origins.min() < 167 or origins.max() >= len(values):
        raise ValueError("Each origin requires 168 observed history rows")
    if static.values.shape != (values.shape[1], 4):
        raise ValueError("Static statistics must match the beam axis")
    phases = origins if phase_origins is None else np.asarray(phase_origins)
    if phases.shape != origins.shape:
        raise ValueError("One absolute phase origin is required per origin")
    # Exclude future observations even from intermediate cumulative sums.
    observed = values[:origins.max() + 1].astype(np.float64, copy=False)
    target = observed[:, :, 0]
    columns = [target[origins - lag] for lag in LAGS]
    columns.extend(target[origins + horizon - 24 * day] for day in (1, 2, 3, 7))
    seasonal = np.stack([target[origins + horizon - 24 * day] for day in range(1, 8)])
    count = np.isfinite(seasonal).sum(0)
    columns.append(np.divide(np.nansum(seasonal, axis=0), count,
                             out=np.full(count.shape, np.nan), where=count > 0))

    def rolling(kpi: int, window: int, full: bool = False) -> list[np.ndarray]:
        data = observed[:, :, kpi]
        means, stds, maxima = [], [], []
        # Compute each actual window identically in batch and single-origin inference.
        # Prefix subtraction can turn an exactly-zero window into tiny nonzero values.
        for t in origins:
            block = data[t-window+1:t+1]
            finite = np.isfinite(block)
            n = finite.sum(0)
            mean = np.divide(np.where(finite, block, 0).sum(0), n,
                             out=np.full(n.shape, np.nan), where=n > 0)
            means.append(mean)
            if full:
                variance = np.divide(np.where(finite, (block - mean) ** 2, 0).sum(0), n,
                                     out=np.full(n.shape, np.nan), where=n > 0)
                maximum = np.max(np.where(finite, block, -np.inf), axis=0)
                maximum[n == 0] = np.nan
                stds.append(np.sqrt(variance))
                maxima.append(maximum)
        return [np.stack(means), np.stack(stds), np.stack(maxima)] if full else [np.stack(means)]

    for window in (6, 24, 168):
        columns.extend(rolling(0, window, full=True))
    for period in (24, 168):
        columns.append(np.broadcast_to(((phases + horizon) % period)[:, None], count.shape))
    for kpi in range(1, 4):
        columns.extend(observed[origins - lag, :, kpi] for lag in (0, 1, 24))
        columns.extend(rolling(kpi, 6))
        columns.extend(rolling(kpi, 24))
    columns.extend(np.broadcast_to(static.values[:, i], count.shape) for i in range(4))
    return np.stack(columns, axis=-1).reshape(-1, len(FEATURE_NAMES)).astype(np.float32)
