"""Topology groups and causal full-cell batches; no fitted per-beam summaries."""

import numpy as np
import torch

from src.data.schema import parse_beam_id


def cell_groups(beam_ids):
    groups = {}
    for index, name in enumerate(beam_ids):
        station, cell, beam = parse_beam_id(name)
        groups.setdefault((station, cell), []).append((beam, index))
    result = []
    for key in sorted(groups):
        items = sorted(groups[key])
        if len(items) != 32 or len({b for b, _ in items}) != 32:
            raise ValueError("Each cell must contain exactly 32 distinct beams")
        result.append([index for _, index in items])
    if len({name for name in beam_ids}) != len(beam_ids):
        raise ValueError("Duplicate beam identity")
    return np.asarray(result, dtype=np.int64)


class CellBatches:
    """Yield a whole cell per sample; future data enters labels only."""

    def __init__(self, panel, preprocessor, origins, device="cpu"):
        if panel.beam_ids != preprocessor.beam_ids or preprocessor.variant != "B":
            raise ValueError("Aligned four-KPI preprocessing required")
        self.origins = np.asarray(origins, dtype=np.int64)
        if not len(self.origins) or self.origins.min() < 167 or self.origins.max()+24 >= len(panel.time):
            raise ValueError("Complete context and joint labels required")
        self.groups = cell_groups(panel.beam_ids)
        self.device = device
        self.values = torch.as_tensor(preprocessor.transform(panel.values, "B"), device=device)
        self.target = torch.as_tensor(panel.values[:, :, 0].astype(np.float32), device=device)
        self.preprocessor = preprocessor
        self.pairs = np.array(np.meshgrid(self.origins, np.arange(len(self.groups)), indexing="ij")).reshape(2, -1).T
        self.n_samples = len(self.pairs)

    def batches(self, batch_size, shuffle_seed=None):
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
            target = self.target[origins[:, None, None]+horizons[None, None, :], beams[:, :, None]]
            yield history, phase, target
