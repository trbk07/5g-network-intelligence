"""Causal research batches retaining short-horizon labels near a frozen cutoff."""

import numpy as np
import torch

from src.features.cell_inputs import cell_groups


class PartialCellBatches:
    """Store only the explicit label prefix; unavailable future labels stay NaN.

    This separate batcher does not change historical joint-label experiments.
    Labels use physical DLPRB units; inputs retain the existing phase4/mask
    and global per-KPI preprocessing contract.
    """

    def __init__(self, panel, preprocessor, origins, label_end: int, device: str = "cpu"):
        origins = np.asarray(origins)
        if (not isinstance(label_end, (int, np.integer)) or isinstance(label_end, bool)
                or not 167 <= label_end < len(panel.time)):
            raise ValueError("Explicit in-panel integer label cutoff required")
        if (origins.ndim != 1 or not len(origins) or not np.issubdtype(origins.dtype, np.integer)
                or origins.min() < 167 or origins.max() >= label_end or np.any(np.diff(origins) <= 0)):
            raise ValueError("Ordered unique integer origins with history and at least one label required")
        if (panel.beam_ids != preprocessor.beam_ids or preprocessor.variant != "B"
                or preprocessor.fit_end > label_end):
            raise ValueError("Aligned fit-prefix four-KPI preprocessing required")
        self.origins = origins.astype(np.int64)
        self.label_end, self.device = int(label_end), device
        self.groups, self.preprocessor = cell_groups(panel.beam_ids), preprocessor
        # Crop before transforming or gathering, so future metric values never
        # enter either backing tensor, including temporarily before masking.
        prefix = panel.values[:label_end+1]
        self.values = torch.as_tensor(preprocessor.transform(prefix, "B"), device=device)
        self.target = torch.as_tensor(prefix[:, :, 0].astype(np.float32), device=device)
        self.pairs = np.array(np.meshgrid(self.origins, np.arange(len(self.groups)), indexing="ij")).reshape(2, -1).T
        self.n_samples = len(self.pairs)

    def batches(self, batch_size: int, shuffle_seed: int | None = None):
        """Yield complete histories and only labels available by label_end."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("Positive integer batch size required")
        order = np.arange(self.n_samples)
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        offsets = torch.arange(-167, 1, device=self.device)
        horizons = torch.tensor([1, 6, 24], device=self.device)
        for start in range(0, len(order), batch_size):
            pairs = self.pairs[order[start:start+batch_size]]
            origins = torch.as_tensor(pairs[:, 0], device=self.device)
            beams = torch.as_tensor(self.groups[pairs[:, 1]], device=self.device)
            history = self.values[origins[:, None, None]+offsets[None, :, None], beams[:, None, :]]
            phase = torch.as_tensor(self.preprocessor.static_at(pairs[:, 0], pairs[:, 1]), device=self.device)
            times = origins[:, None, None]+horizons[None, None, :]
            available = times <= self.label_end
            # Out-of-prefix indices are never gathered. The endpoint is merely
            # a safe temporary index; its values are replaced with missing labels.
            gathered = self.target[times.clamp_max(self.label_end), beams[:, :, None]]
            target = torch.where(available, gathered, torch.full_like(gathered, float("nan")))
            yield history, phase, target
