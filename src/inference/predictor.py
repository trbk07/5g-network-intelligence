"""Shared API/dashboard inference: immutable artifacts, history-only features, JSON-safe output."""

from threading import Lock

import numpy as np

from src.inference.registry import load_champion
from src.inference.replay_store import ReplayStore
from src.risk.scorer import risk_status
from src.risk.thresholds import Thresholds


def number(value):
    """JSON has no NaN/Infinity; preserve missingness as null."""
    return float(value) if np.isfinite(value) else None


class Predictor:
    """One initialized instance per API process, serialized native model calls."""

    def __init__(self, directory="demo_assets"):
        self.store = ReplayStore(directory)
        self.models, self.thresholds, self.metadata = load_champion(
            self.store.directory / "champion"
        )
        if self.store.manifest["frozen_config_sha256"] != self.metadata["frozen_config_sha256"]:
            raise ValueError("Demo/model frozen configuration mismatch")
        if self.store.manifest["dataset_revision"] != self.metadata["dataset_revision"]:
            raise ValueError("Demo/model dataset revision mismatch")
        if self.store.min_origin != self.metadata["fit_end"]:
            raise ValueError("Demo/model fit boundary mismatch")
        self.beam_index = {b: i for i, b in enumerate(self.thresholds.beam_ids)}
        if not set(self.store.index).issubset(self.beam_index):
            raise ValueError("Unknown replay beam in champion context")
        self.version = self.metadata["freeze_commit"][:12]
        self.lock = Lock()

    def info(self):
        """Product-visible provenance and measured limitations, no fake live claims."""
        return {
            "champion": self.metadata["family"],
            "model_version": self.version,
            "horizons": [1, 6, 24],
            "dataset_revision": self.metadata["dataset_revision"],
            "is_replay": True,
            "forecast_quantiles_available": False,
            "preprocessing": "40 causal inputs; native missing values",
            "limitations": [
                "Historical offline replay, not live telemetry.",
                "P90/P95 are historical beam thresholds, not forecast quantiles or congestion probabilities.",
                "Measured P90 recall is only about 10-16%; many exceedances are missed.",
                "Internal one-week holdout; external future-network generalization is unproven.",
            ],
        }

    def predict(self, beam_ids, origin):
        """Predict first; only then attach known actuals for display/parity."""
        history = self.store.history(beam_ids, origin)
        forecasts = {}
        with self.lock:
            for h, model in self.models.items():
                predicted = (
                    model.predict_at(history, origin, h)
                    if hasattr(model, "predict_at")
                    else model.predict(history, h)
                )
                predicted = np.asarray(predicted)
                if predicted.shape != (len(beam_ids),) or np.isinf(predicted).any():
                    raise ValueError("Invalid model output")
                forecasts[h] = predicted
        beams = []
        for j, beam in enumerate(beam_ids):
            missing_history = not np.isfinite(history[:, j, 0]).any()
            beams.append(
                {
                    "beam_id": beam,
                    "history": [
                        {"time": origin - 167 + i, "value": number(v)}
                        for i, v in enumerate(history[:, j, 0])
                    ],
                    "predictions": [
                        {
                            "horizon": h,
                            "target_index": origin + h,
                            "point": number(forecasts[h][j]),
                            "actual": number(self.store.actual(beam_ids, origin + h)[j]),
                            "flags": (["missing_history"] if missing_history else [])
                            + ([] if np.isfinite(forecasts[h][j]) else ["missing_prediction"]),
                        }
                        for h in (1, 6, 24)
                    ],
                }
            )
        return {"origin": origin, "model_version": self.version, "is_replay": True, "beams": beams}

    def predict_risk(self, beam_ids, origin):
        """Score the exact same forecasts with frozen historical P90/P95 guards."""
        forecast = self.predict(beam_ids, origin)
        indices = [self.beam_index[b] for b in beam_ids]
        t = self.thresholds
        subset = Thresholds(
            t.warning[indices],
            t.critical[indices],
            t.valid[indices],
            t.reasons[indices],
            t.active_ratio[indices],
            t.n_observed[indices],
            tuple(beam_ids),
            t.fit_end,
        )
        points = np.array(
            [
                [
                    b["predictions"][i]["point"]
                    if b["predictions"][i]["point"] is not None
                    else np.nan
                    for b in forecast["beams"]
                ]
                for i in range(3)
            ]
        )
        statuses = risk_status(points, subset)
        for j, beam in enumerate(forecast["beams"]):
            beam["historical_warning_p90"] = number(subset.warning[j])
            beam["historical_critical_p95"] = number(subset.critical[j])
            beam["threshold_fit_end"] = subset.fit_end
            for i, prediction in enumerate(beam["predictions"]):
                missing_history = "missing_history" in prediction["flags"]
                valid = bool(subset.valid[j] and np.isfinite(points[i, j]) and not missing_history)
                prediction.update(
                    status="Excluded" if missing_history else str(statuses[i, j]),
                    valid_mask=valid,
                    reason=None
                    if valid
                    else (
                        "missing_history"
                        if missing_history
                        else "missing_prediction"
                        if not np.isfinite(points[i, j])
                        else str(subset.reasons[j])
                    ),
                )
        return forecast
