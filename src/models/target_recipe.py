"""Research-only target recipes with the original 40 causal numerical inputs."""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.features.lightgbm_features import FEATURE_NAMES
from src.utils.runtime import require_cloud_training

SCHEMA = "lgbm-causal40-targetrecipe-v1"


def components(x, transform):
    """Only already observed input columns define an anchor and dynamic scale."""

    def column(name):
        return x[:, FEATURE_NAMES.index(name)].astype(float)

    anchor = np.zeros(len(x))
    scale = np.ones(len(x))
    if transform == "residual_last":
        anchor = column("DLPRB_lag_0")
    elif transform in ("residual_seasonal", "normalized_residual"):
        anchor = column("DLPRB_target_seasonal_mean_7d")
    elif transform not in ("direct", "normalized"):
        raise ValueError("Unknown target recipe")
    if transform in ("normalized", "normalized_residual"):
        scale = np.maximum(np.nan_to_num(column("DLPRB_roll_168_mean"), nan=0.0), 0.1)
    anchor = np.nan_to_num(anchor, nan=0.0)
    return anchor, scale


def decode(raw, x, transform):
    """Return original-unit, nonnegative usage; clipping policy fixed before research scores."""
    anchor, scale = components(x, transform)
    return np.maximum(0, anchor + scale * np.asarray(raw))


@dataclass
class RecipeModel:
    booster: lgb.Booster
    recipe: dict
    metadata: dict

    def predict(self, x):
        if x.ndim != 2 or x.shape[1] != 40:
            raise ValueError("Exactly 40 causal inputs required")
        raw = self.booster.predict(x, num_threads=self.metadata["threads"])
        return decode(raw, x, self.recipe["transform"])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(
            str(directory / "model.txt"), num_iteration=self.booster.best_iteration
        )
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "feature_names": FEATURE_NAMES,
                    "recipe": self.recipe,
                    "metadata": self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        if (
            meta["schema"] != SCHEMA
            or meta["feature_names"] != FEATURE_NAMES
            or booster.feature_name() != FEATURE_NAMES
        ):
            raise ValueError("Research target recipe schema mismatch")
        return cls(booster, meta["recipe"], meta["metadata"])


def train_recipe(
    x,
    y,
    stop_x,
    stop_y,
    recipe,
    *,
    seed=17,
    threads=4,
    max_rounds=1000,
    patience=70,
    origins=None,
    fit_end=None,
    fixed_rounds=None,
    metadata=None,
):
    """Cloud only. Early stopping uses original-unit MAE on a separate stopping segment."""
    require_cloud_training()
    if x.shape[1] != 40 or (np.isfinite(y) & (y < 0)).any():
        raise ValueError("Expected causal40 and nonnegative observed usage")
    anchor, scale = components(x, recipe["transform"])
    valid = np.isfinite(y)
    target = (y - anchor) / scale
    # Scale weighting exactly restores original-unit absolute loss under target normalization.
    weights = scale.copy()
    if recipe.get("half_life"):
        if origins is None or fit_end is None or len(origins) != len(y):
            raise ValueError("Recency weighting requires aligned training origins and cutoff")
        weights *= np.exp2((np.asarray(origins) - fit_end) / recipe["half_life"])
    params = {
        "objective": "regression_l1",
        "metric": "None",
        "verbosity": -1,
        "seed": seed,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        **recipe["params"],
    }
    train = lgb.Dataset(
        x[valid], label=target[valid], weight=weights[valid], feature_name=FEATURE_NAMES
    )
    kwargs = {}
    if fixed_rounds is None:
        stop_valid = np.isfinite(stop_y)
        sx, sy = stop_x[stop_valid], stop_y[stop_valid]
        stop_anchor, stop_scale = components(sx, recipe["transform"])
        stopping = lgb.Dataset(sx, label=(sy - stop_anchor) / stop_scale, reference=train)

        def original_mae(prediction, dataset):
            return (
                "original_mae",
                float(np.abs(decode(prediction, sx, recipe["transform"]) - sy).mean()),
                False,
            )

        kwargs = {
            "valid_sets": [stopping],
            "valid_names": ["stopping_only"],
            "feval": original_mae,
            "callbacks": [lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
        }
    booster = lgb.train(params, train, num_boost_round=fixed_rounds or max_rounds, **kwargs)
    meta = {
        **(metadata or {}),
        "seed": seed,
        "threads": threads,
        "best_iteration": booster.best_iteration if fixed_rounds is None else fixed_rounds,
        "research_only": True,
        "feature_count": 40,
        "lightgbm_version": lgb.__version__,
        "stopping_metric": "original_unit_mae",
        "target_weighting": "dynamic_scale restores original-unit L1; optional recency",
        "holdout_used": False,
    }
    return RecipeModel(booster, recipe, meta)
