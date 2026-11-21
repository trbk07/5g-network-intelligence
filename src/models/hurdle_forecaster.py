"""Research-only CPU inference for verified fixed-refit hurdle artifacts."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.io import sha256
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.models.hurdle import mixture_median


class HurdleForecaster:
    """Load a research checkpoint without changing production registry/routing."""

    def __init__(self, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if sha256(root / name) != digest:
                raise ValueError("Hurdle artifact checksum mismatch")
        self.models, self.metadata = {}, {}
        for h in (1, 6, 24):
            folder = root / f"h{h}"
            meta = json.loads((folder / "meta.json").read_text())
            if (meta["horizon"] != h or meta["fit_end"] != 503 or meta["holdout_used"]
                    or not meta["research_only"]):
                raise ValueError("Unexpected hurdle temporal/research contract")
            levels = meta["protocol"]["conditional_quantiles"]
            expected = {"active", "direct"} | {f"q{a:g}" for a in levels}
            if set(meta["rounds"]) != expected:
                raise ValueError("Incomplete hurdle component set")
            models = {name: lgb.Booster(model_file=str(folder / f"{name}.txt")) for name in expected}
            if any(m.num_feature() != 40 or m.feature_name() != FEATURE_NAMES for m in models.values()):
                raise ValueError("Hurdle models require original 40 causal inputs")
            self.metadata[h], self.models[h] = meta, models

    def predict_at(self, history, origin, horizon, model="hurdle_median"):
        history = np.asarray(history)
        if (history.ndim != 3 or history.shape[0] != 168 or history.shape[2] != 4
                or not isinstance(origin, (int, np.integer)) or origin < 167
                or horizon not in self.models or model not in ("hurdle_median", "hurdle_direct_l1")):
            raise ValueError("Expected 168 historical rows, absolute origin and supported research model")
        features = build_features(history, np.array([167]), horizon, phase_origins=np.array([origin]))
        models = self.models[horizon]
        if model == "hurdle_direct_l1":
            return np.maximum(0, models["direct"].predict(features, num_threads=4))
        levels = self.metadata[horizon]["protocol"]["conditional_quantiles"]
        probability = models["active"].predict(features, num_threads=4)
        conditional = np.column_stack([models[f"q{a:g}"].predict(features, num_threads=4) for a in levels])
        return mixture_median(probability, conditional, levels)
