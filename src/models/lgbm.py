"""Direct global LightGBM models with training-only static artifacts."""
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.features.static_stats import StaticStats
from src.utils.runtime import require_cloud_training


@dataclass
class LGBMForecaster:
    """One direct horizon model; absolute origin is required for phase features."""

    booster: lgb.Booster
    static: StaticStats
    horizon: int
    name: str
    metadata: dict

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Reject missing absolute origin instead of silently assigning a false phase."""
        raise ValueError("LightGBM requires predict_at(history, origin, horizon)")

    def predict_at(self, history: np.ndarray, origin: int, horizon: int) -> np.ndarray:
        """Forecast from a local observed window plus a non-observation time index."""
        if horizon != self.horizon:
            raise ValueError("Requested horizon differs from model horizon")
        if origin < self.static.fit_end:
            raise ValueError("Prediction origin precedes model fit boundary")
        features = build_features(history, np.array([len(history) - 1]), horizon, self.static,
                                   phase_origins=np.array([origin]))
        return self.booster.predict(features, num_iteration=self.booster.best_iteration,
                                     num_threads=self.metadata.get("num_threads", 4))


def train_lgbm(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray,
               val_y: np.ndarray, config: dict, static: StaticStats, horizon: int,
               name: str, seed: int = 17, num_threads: int = 4,
               metadata: dict | None = None) -> LGBMForecaster:
    """Fit a direct model; early stop on validation MAE and count missing targets."""
    require_cloud_training()
    train_mask, val_mask = np.isfinite(train_y), np.isfinite(val_y)
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Training and validation require observed targets")
    params = {k: v for k, v in config.items() if k not in ("n_estimators", "early_stopping_rounds")}
    params.update(metric="mae", seed=seed, num_threads=num_threads, verbosity=-1,
                  deterministic=True, force_col_wise=True)
    train_data = lgb.Dataset(train_x[train_mask], label=train_y[train_mask],
                             feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(val_x[val_mask], label=val_y[val_mask], reference=train_data,
                           feature_name=FEATURE_NAMES)
    history = {}
    booster = lgb.train(params, train_data, num_boost_round=config["n_estimators"],
                        valid_sets=[val_data], valid_names=["validation"], callbacks=[
                            lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                            lgb.record_evaluation(history)])
    meta = {**(metadata or {}), "run_config": (metadata or {}).get("config"),
            "config": config, "seed": seed, "num_threads": num_threads,
            "horizon": horizon, "name": name, "feature_names": FEATURE_NAMES,
            "best_iteration": booster.best_iteration, "fit_end": static.fit_end,
            "train_rows": int(train_mask.sum()), "validation_rows": int(val_mask.sum()),
            "train_missing_targets": int((~train_mask).sum()),
            "validation_missing_targets": int((~val_mask).sum()), "history": history,
            "lightgbm_version": lgb.__version__}
    return LGBMForecaster(booster, static, horizon, name, meta)


def save_lgbm(model: LGBMForecaster, directory: str | Path) -> None:
    """Persist text model, JSON metadata and pickle-free per-beam statistics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(directory / "model.txt"), num_iteration=model.booster.best_iteration)
    (directory / "meta.json").write_text(json.dumps(model.metadata, indent=2), encoding="utf-8")
    np.savez_compressed(directory / "static.npz", values=model.static.values,
                        beam_ids=np.array(model.static.beam_ids), fit_end=model.static.fit_end)


def load_lgbm(directory: str | Path) -> LGBMForecaster:
    """Load an independent model and validate the feature/static schema."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    if meta["feature_names"] != FEATURE_NAMES or booster.feature_name() != FEATURE_NAMES:
        raise ValueError("Artifact feature schema mismatch")
    with np.load(directory / "static.npz", allow_pickle=False) as data:
        static = StaticStats(data["values"], tuple(data["beam_ids"].tolist()), int(data["fit_end"]))
    if static.fit_end != meta["fit_end"] or static.values.shape != (len(static.beam_ids), 4):
        raise ValueError("Artifact static schema mismatch")
    return LGBMForecaster(booster, static, meta["horizon"], meta["name"], meta)
