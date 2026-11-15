"""Limited GRU-A/B forecasting architecture from the execution plan."""
import torch
from torch import nn


class GRUNet(nn.Module):
    """Encode 168-hour history, concatenate static6, and forecast three horizons."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = .1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden_size+6, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Map (batch,time,channels) and (batch,6) to (batch,3)."""
        _, hidden = self.gru(sequence)
        return self.head(torch.cat([hidden[-1], static], dim=1))
