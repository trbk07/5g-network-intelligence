"""Research-only CPU inference from hosted temporal-linear checkpoints."""

import json
from pathlib import Path

import numpy as np
import torch

from src.data.io import sha256
from src.features.gru_inputs import GRUPreprocessor
from src.models.temporal_linear import TemporalLinear


class TemporalLinearForecaster:
    """History-only inference; never registers this research model in production."""

    def __init__(self, root, fold_name, variant):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if sha256(root / name) != digest:
                raise ValueError("Temporal-linear checkpoint checksum mismatch")
        if variant not in ("A", "B"):
            raise ValueError("Unsupported temporal-linear variant")
        directory = root / fold_name / variant
        meta = json.loads((directory / "meta.json").read_text())
        if (meta["fold"]["name"] != fold_name or meta["variant"] != variant
                or meta["holdout_used"] or meta["validation_504_671_used"]
                or meta["protocol"]["protocol"] != "temporal-linear-development-v1"
                or not 167 < meta["fold"]["fit_end"] < meta["fold"]["stop_end"] < meta["fold"]["score_end"] <= 503):
            raise ValueError("Unexpected research temporal contract")
        self.preprocessor = GRUPreprocessor.load(directory / "preprocessor.npz")
        if self.preprocessor.fit_end != meta["fold"]["fit_end"] or self.preprocessor.variant != variant:
            raise ValueError("Preprocessing was fitted on a different prefix")
        state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        expected = {"architecture": "temporal-linear-v1", "fold": meta["fold"], "variant": variant}
        if (state["identity"]["run_identity"] != expected
                or state["identity"]["seed"] != meta["protocol"]["seed"]
                or state["best_epoch"] != meta["training"]["best_epoch"]):
            raise ValueError("Checkpoint provenance/epoch differs")
        self.model = TemporalLinear(1 if variant == "A" else 4).eval()
        self.model.load_state_dict(state["model"], strict=True)
        self.metadata = meta

    def predict_at(self, history, origin, horizon):
        history = np.asarray(history)
        if (history.ndim != 3 or history.shape[0] != 168 or history.shape[2] != 4
                or not history.shape[1] or np.isinf(history).any()
                or not isinstance(origin, (int, np.integer)) or origin < 167
                or horizon not in (1, 6, 24)):
            raise ValueError("Expected 168 historical rows, nonempty beams and supported absolute origin/horizon")
        sequence = self.preprocessor.transform(history, self.preprocessor.variant).transpose(1, 0, 2)
        phase = self.preprocessor.static_at(np.full(history.shape[1], origin), np.arange(history.shape[1]))
        with torch.inference_mode():
            output = self.model(torch.as_tensor(sequence), torch.as_tensor(phase))[:, (1,6,24).index(horizon)].numpy()
        return np.maximum(0, output*self.preprocessor.scale[0]+self.preprocessor.mean[0])
