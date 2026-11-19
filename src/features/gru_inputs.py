"""Training-fitted imputation/scaling and causal GRU batches without window materialization."""
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch

from src.data.panel import Panel

SCHEMA = "gru-global-kpi-phase4-v2"


def fit_imputer(panel: Panel, fit_end: int, variant: str = "B") -> np.ndarray:
    """Fit one global median per selected KPI, with no beam dimension."""
    if variant not in ("A", "B") or not 0 <= fit_end < len(panel.time):
        raise ValueError("Invalid variant or fit cutoff")
    result = []
    for k in range(1 if variant == "A" else 4):
        values = panel.values[:fit_end+1, :, k].reshape(-1)
        finite = values[np.isfinite(values)]
        result.append(float(np.median(finite)) if finite.size else 0.)
    return np.asarray(result)


def fit_scaler(panel: Panel, fit_end: int, median: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit global per-KPI mean/std after training-median imputation, not per beam."""
    if not 0 <= fit_end < len(panel.time) or median.shape not in ((1,), (4,)):
        raise ValueError("Invalid cutoff or KPI parameter shape")
    means, scales = [], []
    for k, fill in enumerate(median):
        values = panel.values[:fit_end+1, :, k].reshape(-1)
        imputed = np.where(np.isfinite(values), values, fill)
        means.append(float(imputed.mean()))
        spread = float(imputed.std(ddof=0))
        scales.append(spread if spread >= 1e-8 else 1.)
    return np.asarray(means), np.asarray(scales)


@dataclass
class GRUPreprocessor:
    """Global per-KPI parameters; beam IDs are metadata, never fitted beam parameters."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    beam_ids: tuple[str, ...]
    fit_end: int
    variant: str = "B"

    @classmethod
    def fit(cls, panel: Panel, fit_end: int, variant: str = "B") -> "GRUPreprocessor":
        """Fit only the training prefix; entirely missing KPIs use zero/one fallback."""
        median = fit_imputer(panel, fit_end, variant)
        mean, scale = fit_scaler(panel, fit_end, median)
        return cls(median, mean, scale, panel.beam_ids, fit_end, variant)

    def transform(self, values: np.ndarray, variant: str) -> np.ndarray:
        """Return scaled KPI channels followed by 1=missing masks (A:2, B:8 channels)."""
        if variant != self.variant or values.shape[-1] != 4:
            raise ValueError("Expected variant A/B and four KPI channels")
        values = values[..., :len(self.mean)]
        missing = ~np.isfinite(values)
        # Zero is observed. Only nonfinite entries are replaced; masks retain that information.
        scaled = (np.where(missing, self.median, values) - self.mean) / self.scale
        return np.concatenate([scaled, missing], axis=-1).astype(np.float32)

    def static_at(self, origins: np.ndarray, beams: np.ndarray) -> np.ndarray:
        """Return only sin/cos phases; beam identity/statistics cannot change these four values."""
        columns = []
        for period in (24, 168):
            angle = 2 * np.pi * origins / period
            columns.append(np.stack([np.sin(angle), np.cos(angle)], axis=-1))
        return np.concatenate(columns, axis=-1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Persist numerical preprocessing state without pickle."""
        np.savez_compressed(path, median=self.median, mean=self.mean, scale=self.scale,
                            beam_ids=np.array(self.beam_ids), fit_end=self.fit_end,
                            variant=np.array(self.variant), schema=np.array(SCHEMA))

    @classmethod
    def load(cls, path: str | Path) -> "GRUPreprocessor":
        """Restore a trusted schema from a pickle-free numerical artifact."""
        with np.load(path, allow_pickle=False) as data:
            if "schema" not in data or str(data["schema"]) != SCHEMA or "static" in data:
                raise ValueError("Legacy/incompatible preprocessing artifact: retrain required")
            variant = str(data["variant"])
            expected = 1 if variant == "A" else 4
            if any(data[k].shape != (expected,) for k in ("median", "mean", "scale")):
                raise ValueError("Preprocessing parameters must be global per-KPI vectors")
            return cls(data["median"], data["mean"], data["scale"],
                       tuple(data["beam_ids"].tolist()), int(data["fit_end"]), variant)


class GRUBatches:
    """Keep compact panel tensors on the selected device and assemble windows per batch."""

    def __init__(self, panel: Panel, preprocessor: GRUPreprocessor, variant: str,
                 origins: np.ndarray, beam_indices: np.ndarray, device: str = "cpu") -> None:
        if panel.beam_ids != preprocessor.beam_ids:
            raise ValueError("Preprocessor beam order differs")
        self.origins, self.beams = np.asarray(origins), np.asarray(beam_indices)
        if (not self.origins.size or self.origins.min() < 167
                or self.origins.max() + 24 >= len(panel.time)):
            raise ValueError("Joint origins require 168 history rows and all three future targets")
        self.device = device
        self.values = torch.as_tensor(preprocessor.transform(panel.values, variant), device=device)
        self.target = torch.as_tensor(((panel.values[:, :, 0] - preprocessor.mean[0]) /
                                       preprocessor.scale[0]).astype(np.float32), device=device)
        self._preprocessor = preprocessor
        self.pairs = np.array(np.meshgrid(self.origins, self.beams, indexing="ij")).reshape(2, -1).T
        self.n_samples = len(self.pairs)

    def batches(self, batch_size: int, shuffle_seed: int | None = None
                ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (history, phase4, target3) tensors; NaN labels remain masked."""
        order = np.arange(self.n_samples)
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        offsets = torch.arange(-167, 1, device=self.device)
        horizons = torch.tensor([1, 6, 24], device=self.device)
        for start in range(0, self.n_samples, batch_size):
            pairs = self.pairs[order[start:start+batch_size]]
            origins = torch.as_tensor(pairs[:, 0], device=self.device)
            beams = torch.as_tensor(pairs[:, 1], device=self.device)
            # For origin t, index exactly t-167..t. No target/future value enters the sequence.
            sequence = self.values[origins[:, None] + offsets, beams[:, None]]
            static = torch.as_tensor(self._preprocessor.static_at(pairs[:, 0], pairs[:, 1]),
                                     device=self.device)
            target = self.target[origins[:, None] + horizons, beams[:, None]]
            yield sequence, static, target
