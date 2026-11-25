"""Point-MAE objectives on the unchanged multivariate temporal encoder."""

import math

import torch
from torch.nn import functional as F

from src.models.neural_hurdle import NeuralHurdleMLP, conditional_mixture_median


class MAEAlignedMLP(NeuralHurdleMLP):
    """Separate nonnegative direct and point-MAE-plus-auxiliary hurdle hypotheses."""

    def __init__(self, kpis=4, hidden_size=64, dropout=.1, target_mean=0.,
                 target_scale=1., mode="direct_clipped", auxiliary_weight=.1):
        if mode not in ("direct_clipped", "hurdle_mae") or not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError("Declared MAE mode and nonnegative auxiliary coefficient required")
        super().__init__(kpis, hidden_size, dropout, target_mean, target_scale,
                         "direct" if mode == "direct_clipped" else "hurdle")
        self.mode, self.auxiliary_weight = mode, auxiliary_weight

    def forward(self, sequence, phase):
        prediction = super().forward(sequence, phase)
        return torch.maximum(prediction, self.zero_scaled) if self.mode == "direct_clipped" else prediction

    def predict_physical(self, sequence, phase):
        if self.mode == "hurdle_mae":
            return super().predict_physical(sequence, phase)
        raw = self.components(sequence, phase)
        return torch.where(raw <= self.zero_scaled, 0.,
                           ((raw*self.target_scale)+self.target_mean).clamp_min(0))

    def training_loss(self, sequence, phase, target):
        valid = torch.isfinite(target)
        n = int(valid.sum())
        values = self.components(sequence, phase)
        if self.mode == "direct_clipped":
            prediction = torch.maximum(values, self.zero_scaled)
            return ((prediction[valid]-target[valid]).abs().mean() if n else prediction.sum()*0), n
        logits, quantiles = values
        if not n:
            return (logits.sum()+quantiles.sum())*0, 0
        median = conditional_mixture_median(logits.sigmoid(), quantiles, self.levels)+self.zero_scaled
        point_mae = (median[valid]-target[valid]).abs().mean()
        positive = valid & (target > self.zero_scaled)
        binary = F.binary_cross_entropy_with_logits(logits[valid], positive[valid].float())
        if positive.any():
            error = (target[positive]-self.zero_scaled).unsqueeze(-1)-quantiles[positive]
            pinball = torch.maximum(self.levels*error, (self.levels-1)*error).mean()
        else:
            pinball = quantiles.sum()*0
        # Keep a fixed auxiliary signal below p=.5, where the point median is constant zero.
        return point_mae+self.auxiliary_weight*(binary+pinball), n
