"""Loadable research ensemble, deliberately separate from the frozen production registry."""

import json
from pathlib import Path

import numpy as np

from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.models.target_recipe import RecipeModel


def baseline_from_features(x, name):
    if name == "persistence":
        return x[:, FEATURE_NAMES.index("DLPRB_lag_0")].astype(float)
    if name == "seasonal24":
        return x[:, FEATURE_NAMES.index("DLPRB_target_seasonal_1d")].astype(float)
    raise ValueError("Unknown baseline")


class ResearchEnsemble:
    """CPU inference with 40 causal features, frozen inner-development weights and bias."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / "ensemble.json").read_text())
        if self.plan["schema"] != "causal40-ensemble-research-v1" or self.plan["fit_end"] != 503:
            raise ValueError("Research ensemble schema/cutoff mismatch")
        if set(self.plan["routes"]) != {"1", "6", "24"}:
            raise ValueError("Exactly three research horizon routes required")
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            from src.data.io import sha256

            manifest = json.loads(manifest_path.read_text())
            for name, digest in manifest["files"].items():
                if sha256(self.directory / name) != digest:
                    raise ValueError("Research artifact checksum mismatch")
        self.models = {}
        for h, route in self.plan["routes"].items():
            weights = [m["weight"] for m in route["members"].values()]
            if (
                not weights
                or min(weights) < 0
                or not np.isclose(sum(weights), 1)
                or not np.isfinite(route["bias"])
            ):
                raise ValueError("Invalid frozen ensemble weights/bias")
            for name, member in route["members"].items():
                key = (int(h), name)
                folder = self.directory / member["path"] if member["path"] else None
                if member["family"] == "lgbm":
                    self.models[key] = RecipeModel.load(folder)
                elif member["family"] == "catboost":
                    from catboost import CatBoostRegressor

                    model = CatBoostRegressor()
                    model.load_model(str(folder / "model.cbm"))
                    self.models[key] = model
                elif member["family"] != "baseline":
                    raise ValueError("Unsupported research member")

    def predict_features(self, x, horizon):
        route = self.plan["routes"][str(horizon)]
        result = np.zeros(len(x), dtype=float)
        for name, member in route["members"].items():
            if member["family"] == "baseline":
                prediction = baseline_from_features(x, name)
            else:
                prediction = self.models[(horizon, name)].predict(x)
            result += member["weight"] * np.maximum(0, prediction)
        return np.maximum(0, result + route["bias"])

    def predict_at(self, history, origin, horizon):
        if origin < self.plan["fit_end"] or horizon not in (1, 6, 24) or history.shape[0] != 168:
            raise ValueError("Invalid research inference origin/horizon/history")
        x = build_features(history, np.array([167]), horizon, phase_origins=np.array([origin]))
        return self.predict_features(x, horizon)
