"""TimeXer-inspired target/covariate attention, independently implemented in PyTorch.

Research basis: Wang et al., NeurIPS 2024, https://arxiv.org/abs/2402.19072.
This adaptation adds masks, phase4 and a seasonal residual, uses global KPI
preprocessing and three direct horizons. It is not the paper's benchmark model.
No upstream implementation is vendored.
"""

import math

import torch
from torch import nn


class CovariateBridge(nn.Module):
    """Temporal target mixing, then a global query over observed KPI tokens."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.target_layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.covariate_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.bridge_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_tokens: torch.Tensor, covariate_tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.target_layer(target_tokens)
        global_token = mixed[:, -1:]
        information, _ = self.covariate_attention(global_token, covariate_tokens, covariate_tokens, need_weights=False)
        updated = self.bridge_norm(global_token+self.dropout(information))
        return torch.cat([mixed[:, :-1], updated], dim=1)


class PastCovariateAttention(nn.Module):
    """Shared beam model using 168 observed hours, four KPI/mask pairs and phase4."""

    def __init__(self, config: dict, use_covariates: bool = True):
        super().__init__()
        width, patch = config["width"], config["patch_length"]
        if patch < 1 or 168 % patch or width % config["heads"] or config["layers"] < 1:
            raise ValueError("Complete non-overlapping patches and valid attention dimensions required")
        self.use_covariates, self.patch_length = use_covariates, patch
        self.target_projection = nn.Linear(2*patch, width)
        self.covariate_projection = nn.Linear(2*168, width)
        self.position = nn.Parameter(torch.zeros(1, 168//patch+1, width))
        self.global_token = nn.Parameter(torch.zeros(1, 1, width))
        # These identify the three KPI types, never beam/entity identities.
        self.covariate_types = nn.Parameter(torch.zeros(1, 3, width))
        self.layers = nn.ModuleList([CovariateBridge(width, config["heads"], config["dropout"])
                                    for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear((168//patch+1)*width+4, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("seasonal_indices", torch.tensor([[167+h-24*d for d in range(1, 8)] for h in (1, 6, 24)]))
        self.register_buffer("target_mean", torch.tensor(0.))
        self.register_buffer("target_scale", torch.tensor(1.))

    def configure_target_scale(self, mean: float, scale: float):
        """Bind the existing fit-only scaler; this does not fit any parameters."""
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Finite global target mean and positive scale required")
        self.target_mean.fill_(mean)
        self.target_scale.fill_(scale)
        return self

    def forward(self, sequence: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1:] != (168, 8) or phase.shape != (len(sequence), 4):
            raise ValueError("Exactly 168 observed four-KPI/mask rows and phase4 required")
        # Center with observed last values only, preserving the project's global
        # per-KPI fit-prefix scaling. No per-beam fitted summary is introduced.
        target = torch.stack([sequence[:, :, 0]-sequence[:, -1:, 0].detach(), sequence[:, :, 4]], dim=-1)
        patches = target.reshape(len(sequence), 168//self.patch_length, 2*self.patch_length)
        tokens = torch.cat([self.target_projection(patches), self.global_token.expand(len(sequence), -1, -1)], dim=1)
        tokens = tokens+self.position
        if self.use_covariates:
            values = sequence[:, :, 1:4]-sequence[:, -1:, 1:4].detach()
            covariates = torch.stack([values, sequence[:, :, 5:8]], dim=-1).permute(0, 2, 1, 3).flatten(2)
        else:
            # Same parameters/initialization, but no covariate values or masks
            # enter this control. Learned constant tokens retain equal capacity.
            covariates = sequence.new_zeros((len(sequence), 3, 2*168))
        covariates = self.covariate_projection(covariates)+self.covariate_types
        for layer in self.layers:
            tokens = layer(tokens, covariates)
        correction = self.head(torch.cat([self.norm(tokens).flatten(1), phase], dim=-1))
        baseline = sequence[:, self.seasonal_indices, 0].mean(-1)
        # Clipping is also active during training/stopping, in standardized units.
        return torch.maximum(baseline+correction, -self.target_mean/self.target_scale)

    def predict_physical(self, sequence: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return (self(sequence, phase)*self.target_scale+self.target_mean).clamp_min(0)
