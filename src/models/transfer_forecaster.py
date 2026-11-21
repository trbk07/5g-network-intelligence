"""Independent CPU inference for research transfer artifacts; never routes production by default."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.features.lightgbm_features import FEATURE_NAMES
from src.features.relative_transfer import relative_features


class TransferForecaster:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "meta.json").read_text())
        if (
            self.metadata["schema"] != "causal40-relative-target-transfer-v1"
            or self.metadata["fit_end"] != 503
            or self.metadata["holdout_used"]
        ):
            raise ValueError("Unsupported or invalid research transfer schema/provenance")
        self.horizon = self.metadata["horizon"]
        self.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        if self.booster.feature_name() != FEATURE_NAMES:
            raise ValueError("Transfer artifact feature ordering differs")

    def predict_at(self, history, origin, horizon):
        if (
            horizon != self.horizon
            or origin < 503
            or history.ndim != 3
            or history.shape[0] != 168
            or history.shape[2] != 4
        ):
            raise ValueError("Transfer origin/horizon/history differs from training contract")
        x, scale = relative_features(
            history, np.array([167]), horizon, phase_origins=np.array([origin])
        )
        return np.maximum(0, self.booster.predict(x, num_threads=4) * scale)
