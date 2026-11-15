"""Baselines consume only a history window ending at the forecast origin."""
from dataclasses import dataclass

import numpy as np


def _validate(history: np.ndarray, horizon: int) -> None:
    if history.ndim != 3 or not len(history) or history.shape[2] != 4:
        raise ValueError("Expected nonempty (time, beam, 4 KPI) history")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")


@dataclass(frozen=True)
class Persistence:
    """Repeat DLPRB at the origin."""

    name: str = "persistence"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return one forecast per beam, preserving missing origin observations."""
        _validate(history, horizon)
        return history[-1, :, 0].copy()


@dataclass(frozen=True)
class SeasonalNaive:
    """Use DLPRB at origin + horizon - period; reject unavailable/future references."""

    period: int = 24

    def __post_init__(self) -> None:
        if not isinstance(self.period, int) or isinstance(self.period, bool) or self.period < 1:
            raise ValueError("period must be a positive integer")

    @property
    def name(self) -> str:
        return f"seasonal_naive_{self.period}"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return causal seasonal forecasts, rejecting negative array indexing."""
        _validate(history, horizon)
        index = len(history) - 1 + horizon - self.period
        if not 0 <= index < len(history):
            raise ValueError("Seasonal reference is outside observed history")
        return history[index, :, 0].copy()
