"""Research adaptation of last-value-normalized temporal linear forecasting."""

import torch
from torch import nn


class TemporalLinear(nn.Module):
    """Global 168-hour projection, missing masks and phase4; no beam parameters.

    Inspired by NLinear (cure-lab/LTSF-Linear), adapted to joint 1/6/24h target
    forecasting, global per-KPI preprocessing and deterministic seasonal start.
    This is not a reproduction of the published benchmark configuration.
    """

    def __init__(self, kpis=1):
        super().__init__()
        if kpis not in (1, 4):
            raise ValueError("One or four physical KPI channels required")
        self.kpis = kpis
        self.projection = nn.Linear(168 * 2 * kpis + 4, 3)
        # A deterministic causal starting point, not a locally fitted model.
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            for head, horizon in enumerate((1, 6, 24)):
                for day in range(1, 8):
                    index = 167 + horizon - 24 * day
                    self.projection.weight[head, index * 2 * kpis] = 1 / 7

    def forward(self, sequence, phase):
        if sequence.ndim != 3 or sequence.shape[1:] != (168, 2*self.kpis) or phase.shape != (sequence.shape[0], 4):
            raise ValueError("Expected (batch,168,KPI+mask) history and phase4")
        last = sequence[:, -1:, :self.kpis].detach()
        centered = torch.cat([sequence[:, :, :self.kpis] - last,
                              sequence[:, :, self.kpis:]], dim=-1)
        inputs = torch.cat([centered.flatten(1), phase], dim=1)
        return self.projection(inputs) + last[:, 0, :1]
