"""Fixed-model rolling-origin evaluator with history-only predictor inputs."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.data.panel import Panel
from src.evaluation import metrics


class Forecaster(Protocol):
    """Predict one DLPRB value per beam from observed history only."""

    name: str

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray: ...


@dataclass
class ForecastRecord:
    """Arrays use (origin, beam), with explicit target indices and model identity."""

    model: str
    horizon: int
    origins: np.ndarray
    beam_ids: tuple[str, ...]
    actual: np.ndarray
    predicted: np.ndarray


def history_at(panel: Panel, origin: int, lookback: int = 168) -> np.ndarray:
    """Return an independent read-only window ending at origin; no future rows."""
    if not isinstance(origin, (int, np.integer)) or isinstance(origin, bool):
        raise ValueError("origin must be an integer")
    if lookback < 1 or origin < lookback - 1 or origin >= len(panel.time):
        raise ValueError("Origin has insufficient history or is out of bounds")
    history = panel.values[origin - lookback + 1:origin + 1].copy()
    history.flags.writeable = False
    return history


def evaluate(forecaster: Forecaster, panel: Panel, origins: np.ndarray,
             horizon: int, lookback: int = 168) -> ForecastRecord:
    """Evaluate fixed-model replay; actuals observed through each origin update history."""
    if not np.array_equal(panel.time, np.arange(len(panel.time))):
        raise ValueError("Evaluation requires contiguous relative hourly indices from zero")
    if hasattr(forecaster, "static") and forecaster.static.beam_ids != panel.beam_ids:
        raise ValueError("Model and panel beam ordering differ")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    origins = np.asarray(origins)
    if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or not origins.size:
        raise ValueError("Expected a nonempty integer origin vector")
    if np.any(np.diff(origins) <= 0) or origins[-1] + horizon >= len(panel.time):
        raise ValueError("Origins must increase uniquely and targets must be in bounds")
    forecasts = []
    for origin in origins:
        history = history_at(panel, int(origin), lookback)
        if hasattr(forecaster, "predict_at"):
            # Time context supplies phase only; observed inputs remain history-only.
            prediction = forecaster.predict_at(history, int(origin), horizon)
        else:
            prediction = forecaster.predict(history, horizon)
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != (len(panel.beam_ids),) or np.isinf(prediction).any():
            raise ValueError("Forecast must be one value per beam, finite or NaN")
        forecasts.append(prediction.copy())
    return ForecastRecord(forecaster.name, horizon, origins.copy(), panel.beam_ids,
                          panel.values[origins + horizon, :, 0].copy(), np.stack(forecasts))


def to_benchmark_rows(record: ForecastRecord, load_tiers: np.ndarray | None = None) -> list[dict]:
    """Summarize pooled and per-load-tier metrics with missingness/coverage counts."""
    groups = {"All": np.ones(len(record.beam_ids), dtype=bool)}
    if load_tiers is not None:
        if np.asarray(load_tiers).shape != (len(record.beam_ids),):
            raise ValueError("Expected one load tier per beam")
        groups.update({tier: load_tiers == tier for tier in ("Low", "Mid", "High", "Unknown")
                       if np.any(load_tiers == tier)})
    functions = {"mae": metrics.mae, "rmse": metrics.rmse, "wape": metrics.wape,
                 "macro_wape": metrics.macro_wape, "macro_mae": metrics.macro_mae,
                 "active_only_mae": metrics.active_only_mae, "zero_fp_rate": metrics.zero_fp_rate}
    rows = []
    for tier, mask in groups.items():
        a, p = record.actual[:, mask], record.predicted[:, mask]
        valid = np.isfinite(a) & np.isfinite(p)
        row = {"model": record.model, "horizon": record.horizon, "load_tier": tier,
               "n_origins": len(record.origins), "n_beams": int(mask.sum()),
               "n_total": a.size, "n_valid": int(valid.sum()),
               "n_excluded": int((~valid).sum()), "n_missing_target": int(np.isnan(a).sum()),
               "n_missing_prediction": int(np.isnan(p).sum()),
               "n_excluded_beams": int((~valid.any(axis=0)).sum()),
               "n_active": int(((a > 0) & valid).sum()),
               "n_zero": int(((a == 0) & valid).sum())}
        for name, function in functions.items():
            result = function(a, p)
            row[name] = result.value
            row[f"{name}_flags"] = "|".join(result.flags)
        rows.append(row)
    return rows


def save_forecasts(records: list[ForecastRecord], path: str | Path,
                    metadata: dict | None = None) -> None:
    """Save variable-length horizon records in a pickle-free compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    index = []
    for i, record in enumerate(records):
        prefix = f"record_{i}"
        index.append({"prefix": prefix, "model": record.model, "horizon": record.horizon})
        for name, value in {"origins": record.origins, "targets": record.origins + record.horizon,
                            "beam_ids": np.array(record.beam_ids), "actual": record.actual,
                            "predicted": record.predicted}.items():
            arrays[f"{prefix}_{name}"] = value
    arrays["metadata"] = np.array(json.dumps({"records": index, **(metadata or {})}))
    np.savez_compressed(path, **arrays)
