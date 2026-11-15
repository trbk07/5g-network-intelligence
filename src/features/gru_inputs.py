"""Training-fitted imputation/scaling and causal GRU batches without window materialization."""
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch

from src.data.panel import Panel
from src.features.static_stats import compute_static


@dataclass
class GRUPreprocessor:
    """Global per-KPI median/mean/std, per-beam normalized static stats and fit boundary."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    static: np.ndarray
    beam_ids: tuple[str, ...]
    fit_end: int

    @classmethod
    def fit(cls, panel: Panel, fit_end: int) -> "GRUPreprocessor":
        """Fit only the training prefix; entirely missing KPIs use zero/one fallback."""
        stats = compute_static(panel, fit_end)
        # Never fit preprocessing on validation: slice the training prefix before any reduction.
        values = panel.values[:fit_end+1]
        median = np.array([np.median(values[:, :, k][np.isfinite(values[:, :, k])])
                           if np.isfinite(values[:, :, k]).any() else 0 for k in range(4)])
        imputed = np.where(np.isfinite(values), values, median)
        mean = imputed.mean(axis=(0, 1))
        scale = imputed.std(axis=(0, 1))
        scale[scale < 1e-8] = 1
        beam_stats = stats.values[:, :2].copy()
        for k in range(2):
            finite = np.isfinite(beam_stats[:, k])
            center = beam_stats[finite, k].mean() if finite.any() else 0
            spread = beam_stats[finite, k].std() if finite.any() else 1
            beam_stats[:, k] = (np.where(finite, beam_stats[:, k], center) - center) / max(spread, 1e-8)
        return cls(median, mean, scale, beam_stats.astype(np.float32), panel.beam_ids, fit_end)

    def transform(self, values: np.ndarray, variant: str) -> np.ndarray:
        """Return scaled KPI channels followed by 1=missing masks (A:2, B:8 channels)."""
        if variant not in ("A", "B") or values.shape[-1] != 4:
            raise ValueError("Expected variant A/B and four KPI channels")
        missing = ~np.isfinite(values)
        # Zero is observed. Only nonfinite entries are replaced; masks retain that information.
        scaled = (np.where(missing, self.median, values) - self.mean) / self.scale
        indices = [0] if variant == "A" else [0, 1, 2, 3]
        return np.concatenate([scaled[..., indices], missing[..., indices]], axis=-1).astype(np.float32)

    def static_at(self, origins: np.ndarray, beams: np.ndarray) -> np.ndarray:
        """Combine beam mean/std with sin/cos of origin phase 24h and 168h."""
        columns = [self.static[beams]]
        for period in (24, 168):
            angle = 2 * np.pi * origins / period
            columns.append(np.stack([np.sin(angle), np.cos(angle)], axis=-1))
        return np.concatenate(columns, axis=-1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Persist numerical preprocessing state without pickle."""
        np.savez_compressed(path, median=self.median, mean=self.mean, scale=self.scale,
                            static=self.static, beam_ids=np.array(self.beam_ids), fit_end=self.fit_end)

    @classmethod
    def load(cls, path: str | Path) -> "GRUPreprocessor":
        """Restore a trusted schema from a pickle-free numerical artifact."""
        with np.load(path, allow_pickle=False) as data:
            return cls(data["median"], data["mean"], data["scale"], data["static"],
                       tuple(data["beam_ids"].tolist()), int(data["fit_end"]))


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
        self.static = torch.as_tensor(preprocessor.static, device=device)
        self._preprocessor = preprocessor
        self.pairs = np.array(np.meshgrid(self.origins, self.beams, indexing="ij")).reshape(2, -1).T
        self.n_samples = len(self.pairs)

    def batches(self, batch_size: int, shuffle_seed: int | None = None
                ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (history, static6, target3) tensors; NaN targets remain masked by the loss."""
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
