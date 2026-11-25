"""Multivariate temporal encoder with matched direct and zero-mass output heads."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def conditional_mixture_median(probability, quantiles, levels):
    """Analytical nonnegative mixture median with fixed quantile interpolation."""
    if probability.shape != quantiles.shape[:-1] or quantiles.shape[-1] != len(levels):
        raise ValueError("Probability and positive-quantile axes differ")
    grid = torch.cat([levels.new_zeros(1), levels])
    values = torch.cat([quantiles.new_zeros((*probability.shape, 1)), quantiles], dim=-1)
    q = torch.where(probability > .5, 1-.5/probability.clamp_min(1e-8), 0.)
    upper = torch.searchsorted(grid, q.contiguous(), right=True).clamp(1, len(grid)-1)
    lower = upper-1
    fraction = (q-grid[lower])/(grid[upper]-grid[lower])
    lo = values.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    hi = values.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    return torch.where(probability > .5, lo+(hi-lo)*fraction, 0.)


class NeuralHurdleMLP(nn.Module):
    """Shared causal inputs; head kind is frozen before any stopping/scoring.

    Inputs include centered history/masks, last global-scaled KPI levels,
    causal seven-day target means and phase4. No entity summaries are fitted.
    Conditional quantiles are expressed in target/scale units, not z scores.
    """

    def __init__(self, kpis=4, hidden_size=64, dropout=.1, target_mean=0.,
                 target_scale=1., kind="hurdle"):
        super().__init__()
        if (kpis not in (1, 4) or hidden_size < 1 or not 0 <= dropout < 1
                or kind not in ("direct", "hurdle") or not math.isfinite(target_mean)
                or not math.isfinite(target_scale) or target_scale <= 0):
            raise ValueError("Valid architecture and global target scaling required")
        self.kpis, self.kind = kpis, kind
        self.encoder = nn.Sequential(nn.Linear(168*2*kpis+kpis+3+4, hidden_size),
                                     nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_size, 3 if kind == "direct" else 15)
        self.register_buffer("zero_scaled", torch.tensor(-target_mean/target_scale, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float64))
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float64))
        self.register_buffer("levels", torch.tensor([.05, .1, .25, .5]))
        self.register_buffer("seasonal_indices", torch.tensor(
            [[167+h-24*d for d in range(1, 8)] for h in (1, 6, 24)]))
        # A deterministic output initialization, with no fitting or calibration.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def components(self, sequence, phase):
        if (sequence.ndim != 3 or sequence.shape[1:] != (168, 2*self.kpis)
                or phase.shape != (len(sequence), 4)):
            raise ValueError("Expected 168-row KPI/mask sequence and phase4")
        last = sequence[:, -1, :self.kpis]
        centered = torch.cat([sequence[:, :, :self.kpis]-last[:, None],
                              sequence[:, :, self.kpis:]], dim=-1)
        baseline = sequence[:, self.seasonal_indices, 0].mean(-1)
        features = torch.cat([centered.flatten(1), last, baseline, phase], dim=1)
        raw = self.head(self.encoder(features))
        if self.kind == "direct":
            return baseline+raw
        raw = raw.reshape(-1, 3, 5)
        # Positive increments enforce noncrossing conditional quantiles by construction.
        return raw[:, :, 0], F.softplus(raw[:, :, 1:]).cumsum(-1)

    def forward(self, sequence, phase):
        values = self.components(sequence, phase)
        if self.kind == "direct":
            return values
        logits, quantiles = values
        positive_scaled = conditional_mixture_median(logits.sigmoid(), quantiles, self.levels)
        return positive_scaled+self.zero_scaled

    def predict_physical(self, sequence, phase):
        """Decode exact zero mass without a z-score inverse-rounding artifact."""
        values = self.components(sequence, phase)
        if self.kind == "direct":
            return (values*self.target_scale+self.target_mean).clamp_min(0)
        logits, quantiles = values
        return conditional_mixture_median(logits.sigmoid(), quantiles, self.levels)*self.target_scale

    def training_loss(self, sequence, phase, target):
        """BCE plus positive-only mean pinball; stopping still uses point MAE."""
        valid = torch.isfinite(target)
        n = int(valid.sum())
        values = self.components(sequence, phase)
        if self.kind == "direct":
            return ((values[valid]-target[valid]).abs().mean() if n else values.sum()*0), n
        logits, quantiles = values
        if not n:
            return (logits.sum()+quantiles.sum())*0, 0
        positive = valid & (target > self.zero_scaled)
        binary = F.binary_cross_entropy_with_logits(logits[valid], positive[valid].float())
        if positive.any():
            error = (target[positive]-self.zero_scaled).unsqueeze(-1)-quantiles[positive]
            pinball = torch.maximum(self.levels*error, (self.levels-1)*error).mean()
        else:
            pinball = quantiles.sum()*0
        return binary+pinball, n
