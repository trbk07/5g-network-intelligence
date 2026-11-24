"""Small dilated causal convolution residual forecaster; no beam summaries."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    """Left-only convolution; normalization is per time step across channels."""

    def __init__(self,channels,dilation,dropout):
        super().__init__()
        self.left_padding=2*dilation
        self.convolution=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.normalization=nn.LayerNorm(channels)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        residual=self.convolution(F.pad(x,(self.left_padding,0)))
        residual=self.normalization(residual.transpose(1,2)).transpose(1,2)
        return x+self.dropout(F.gelu(residual))


class TemporalResidualTCN(nn.Module):
    """Seven dyadic blocks plus phase4 head and fixed daily seasonal mean."""

    def __init__(self,kpis=1,channels=32,levels=7,dropout=.1):
        super().__init__()
        if kpis not in (1,4) or channels<1 or levels<1 or not 0<=dropout<1:
            raise ValueError("Valid KPI count, channels, levels and dropout required")
        self.kpis=kpis
        self.projection=nn.Conv1d(2*kpis,channels,1)
        self.blocks=nn.Sequential(*(CausalResidualBlock(channels,2**i,dropout) for i in range(levels)))
        self.head=nn.Linear(channels+4,3)
        self.register_buffer("seasonal_indices",torch.tensor([[167+h-24*d for d in range(1,8)] for h in (1,6,24)]))
        # Analytical initial forecast only; no fitting during construction.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self,sequence,phase):
        if sequence.ndim!=3 or sequence.shape[1:]!=(168,2*self.kpis) or phase.shape!=(sequence.shape[0],4):
            raise ValueError("Expected 168-row physical-KPI/mask sequence and phase4")
        last=sequence[:,-1:,:self.kpis].detach()
        centered=torch.cat([sequence[:,:,:self.kpis]-last,sequence[:,:,self.kpis:]],dim=-1)
        encoded=self.blocks(self.projection(centered.transpose(1,2)))[:,:,-1]
        baseline=sequence[:,self.seasonal_indices,0].mean(-1)
        return baseline+self.head(torch.cat([encoded,phase],dim=1))
