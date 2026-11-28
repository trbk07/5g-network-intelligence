"""Shared temporal patches with controlled within-cell spatial attention."""

import torch
from torch import nn


class CellPatchAttention(nn.Module):
    def __init__(self, config, mean, scale, mode):
        super().__init__()
        if mode not in ("local", "cell"):
            raise ValueError("Explicit spatial ablation mode required")
        self.mode = mode
        width = config["width"]
        self.patch_length, self.patch_stride = config["patch_length"], config["patch_stride"]
        patches = (168-self.patch_length)//self.patch_stride+1
        self.patch_projection = nn.Linear(8*self.patch_length, width)
        self.position = nn.Parameter(torch.zeros(1, patches, width))
        layer = nn.TransformerEncoderLayer(width, config["heads"], 2*width,
            config["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, config["temporal_layers"], enable_nested_tensor=False)
        self.history_projection = nn.Sequential(nn.Flatten(1), nn.Linear(patches*width, width), nn.GELU())
        self.spatial = nn.MultiheadAttention(width, config["heads"], config["dropout"], batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width+8, width), nn.GELU(), nn.Dropout(config["dropout"]), nn.Linear(width, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("target_mean", torch.tensor(float(mean)))
        self.register_buffer("target_scale", torch.tensor(float(scale)))
        self.register_buffer("seasonal_indices", torch.tensor([[167+h-24*d for d in range(1, 8)] for h in (1, 6, 24)]))
        self.register_buffer("local_mask", ~torch.eye(32, dtype=torch.bool))

    def forward(self, history, phase):
        if history.ndim != 4 or history.shape[1:] != (168, 32, 8) or phase.shape != (len(history), 4):
            raise ValueError("Expected batch x 168 x 32 x 8 causal history and phase4")
        batch = len(history)
        # Only this origin's observed window is centered; no learned beam statistics.
        last = history[:, -1:, :, :4].detach()
        centered = torch.cat([history[:, :, :, :4]-last, history[:, :, :, 4:]], dim=-1)
        channels = centered.permute(0, 2, 3, 1).reshape(batch*32, 8, 168)
        patches = channels.unfold(-1, self.patch_length, self.patch_stride).permute(0, 2, 1, 3).flatten(2)
        temporal = self.temporal(self.patch_projection(patches)+self.position)
        tokens = self.history_projection(temporal).reshape(batch, 32, -1)
        mixed, _ = self.spatial(tokens, tokens, tokens,
            attn_mask=self.local_mask if self.mode == "local" else None, need_weights=False)
        features = self.norm(tokens+mixed)
        own_level = history[:, -1, :, :4]
        phase = phase[:, None, :].expand(-1, 32, -1)
        correction = self.head(torch.cat([features, own_level, phase], dim=-1))
        baseline = history[:, self.seasonal_indices, :, 0].mean(2).permute(0, 2, 1)
        # Train and select directly in physical DLPRB units with nonnegative outputs.
        return ((baseline+correction)*self.target_scale+self.target_mean).clamp_min(0)
