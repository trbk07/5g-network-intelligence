"""Independent GRU artifact inference over observed histories only."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data.panel import Panel

from src.features.gru_inputs import GRUPreprocessor
from src.models.gru import GRUNet


class GRUForecaster:
    """Load one seed/variant; return unscaled predictions for all three horizons."""

    def __init__(self, directory: str | Path, device: str = "cpu") -> None:
        directory = Path(directory)
        self.metadata = json.loads((directory / "meta.json").read_text())
        self.preprocessor = GRUPreprocessor.load(directory / "preprocessor.npz")
        self.device = device
        self.variant = self.metadata["variant"]
        self.name = f"gru_{self.variant}_seed{self.metadata['seed']}"
        self.model = GRUNet(2 if self.variant == "A" else 8, **self.metadata["architecture"]).to(device)
        state = torch.load(directory / "best.pt", map_location=device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def predict_all(self, history: np.ndarray, origin: int, batch_size: int = 256) -> np.ndarray:
        """Return (beam,3) forecasts using exactly the observed 168-hour window."""
        if history.shape != (168, len(self.preprocessor.beam_ids), 4) or origin < self.preprocessor.fit_end:
            raise ValueError("History shape or origin is incompatible with the fitted model")
        sequence = self.preprocessor.transform(history, self.variant).transpose(1, 0, 2)
        static = self.preprocessor.static_at(np.full(len(sequence), origin), np.arange(len(sequence)))
        predictions = []
        with torch.no_grad():
            for start in range(0, len(sequence), batch_size):
                x = torch.as_tensor(sequence[start:start+batch_size], device=self.device)
                s = torch.as_tensor(static[start:start+batch_size], device=self.device)
                predictions.append(self.model(x, s).cpu().numpy())
        scaled = np.concatenate(predictions)
        return scaled * self.preprocessor.scale[0] + self.preprocessor.mean[0]

    def predict_at(self, history: np.ndarray, origin: int, horizon: int) -> np.ndarray:
        """Select one requested horizon with absolute origin phase."""
        if horizon not in (1, 6, 24):
            raise ValueError("Unsupported horizon")
        return self.predict_all(history, origin)[:, (1, 6, 24).index(horizon)]

    def predict_origins(self, panel: Panel, origins: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Batch replay over all beams, returning (origin,beam,horizon) without future inputs."""
        origins = np.asarray(origins)
        if (origins.ndim != 1 or not origins.size or not np.issubdtype(origins.dtype, np.integer)
                or np.any(np.diff(origins) <= 0)
                or origins.min() < max(167, self.preprocessor.fit_end)
                or origins.max() >= len(panel.time)):
            raise ValueError("Origins must have sufficient observed history and valid indices")
        if panel.beam_ids != self.preprocessor.beam_ids:
            raise ValueError("Beam order or forecast origins mismatch")
        values = torch.as_tensor(self.preprocessor.transform(panel.values[:origins.max()+1],
                                                              self.variant), device=self.device)
        beams = np.arange(len(panel.beam_ids))
        pairs = np.array(np.meshgrid(origins, beams, indexing="ij")).reshape(2, -1).T
        offsets = torch.arange(-167, 1, device=self.device)
        result = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start+batch_size]
                t = torch.as_tensor(batch[:, 0], device=self.device)
                b = torch.as_tensor(batch[:, 1], device=self.device)
                history = values[t[:, None] + offsets, b[:, None]]
                static = torch.as_tensor(self.preprocessor.static_at(batch[:, 0], batch[:, 1]),
                                         device=self.device)
                result.append(self.model(history, static).cpu().numpy())
        scaled = np.concatenate(result).reshape(len(origins), len(beams), 3)
        return scaled * self.preprocessor.scale[0] + self.preprocessor.mean[0]

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        raise ValueError("GRU requires absolute origin context via predict_at")
