"""Shared nonlinear residual projection of observed history, without beam statistics."""

import torch
from torch import nn


class TemporalResidualMLP(nn.Module):
    """Causal seven-day baseline plus nonlinear residual; phase4 and global KPI inputs."""

    def __init__(self,kpis=1,hidden_size=64,dropout=0.1):
        super().__init__()
        if kpis not in (1,4) or hidden_size<1 or not 0<=dropout<1:
            raise ValueError("Valid KPI count, hidden width and dropout required")
        self.kpis=kpis
        self.residual=nn.Sequential(nn.Linear(168*2*kpis+4,hidden_size),nn.GELU(),
                                    nn.Dropout(dropout),nn.Linear(hidden_size,3))
        self.register_buffer("seasonal_indices",torch.tensor([[167+h-24*d for d in range(1,8)] for h in (1,6,24)]))
        # An analytical starting point only. No local fitting is performed here.
        with torch.no_grad():
            self.residual[-1].weight.zero_()
            self.residual[-1].bias.zero_()

    def forward(self,sequence,phase):
        if sequence.ndim!=3 or sequence.shape[1:]!=(168,2*self.kpis) or phase.shape!=(sequence.shape[0],4):
            raise ValueError("Expected 168-row physical-KPI/mask sequence and phase4")
        last=sequence[:,-1:,:self.kpis].detach()
        centered=torch.cat([sequence[:,:,:self.kpis]-last,sequence[:,:,self.kpis:]],dim=-1)
        inputs=torch.cat([centered.flatten(1),phase],dim=1)
        baseline=sequence[:,self.seasonal_indices,0].mean(-1)
        return baseline+self.residual(inputs)
