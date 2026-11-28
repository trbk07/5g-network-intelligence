"""Unitless source forecasting head; only temporal encoder weights are transferred."""

import torch
from torch import nn

from src.models.cell_attention import CellPatchAttention

ENCODER_PREFIXES = ("patch_projection.", "position", "temporal.", "history_projection.")


class SourcePatchForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CellPatchAttention(config, 0, 1, "local")
        self.head = nn.Sequential(nn.Linear(config["width"]+5, config["width"]), nn.GELU(),
                                  nn.Linear(config["width"], 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, phase):
        if sequence.shape[1:] != (168, 2):
            raise ValueError("Standardized source value and missing mask required")
        inputs = torch.zeros(len(sequence), 168, 8, device=sequence.device, dtype=sequence.dtype)
        inputs[:, :, 0] = sequence[:, :, 0]
        inputs[:, :, 4] = sequence[:, :, 1]
        inputs[:, :, 5:] = 1  # External sources do not fabricate other original KPIs.
        encoded = self.encoder.encode_history(inputs)
        residual = self.head(torch.cat([encoded, sequence[:, -1, :1], phase], dim=-1))
        baseline = sequence[:, self.encoder.seasonal_indices, 0].mean(-1)
        return baseline+residual


def transfer_encoder(model, pretrained):
    """Exclude source head, physical scales, spatial attention and target output head."""
    selected = {name.removeprefix("encoder."): value for name, value in pretrained.items()
                if name.startswith("encoder.") and name.removeprefix("encoder.").startswith(ENCODER_PREFIXES)}
    expected = {name for name in model.state_dict() if name.startswith(ENCODER_PREFIXES)}
    if set(selected) != expected:
        raise ValueError("Incomplete or incompatible temporal encoder transfer")
    state = model.state_dict()
    state.update(selected)
    model.load_state_dict(state)
    return sorted(selected)
