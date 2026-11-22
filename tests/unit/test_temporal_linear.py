"""Initialized-model forward checks only: no fitting, backward or optimizer calls."""

import pytest
import torch

from src.models.temporal_linear import TemporalLinear


@pytest.mark.parametrize("kpis", [1, 4])
def test_seasonal_initialization_is_an_analytical_causal_average(kpis):
    model = TemporalLinear(kpis).eval()
    sequence = torch.arange(168., dtype=torch.float32)[None, :, None].repeat(2, 1, 2*kpis)
    phase = torch.zeros(2, 4)
    expected = torch.tensor([[sum(167+h-24*d for d in range(1, 8))/7 for h in (1, 6, 24)] for _ in range(2)])
    with torch.no_grad():
        torch.testing.assert_close(model(sequence, phase), expected)
        torch.testing.assert_close(model(sequence[:1], phase[:1]), model(sequence, phase)[:1])


def test_last_value_normalization_preserves_target_shift():
    model = TemporalLinear(4).eval()
    sequence = torch.zeros(1, 168, 8)
    shifted = sequence.clone()
    shifted[:, :, :4] += torch.tensor([3., 4., 5., 6.])
    with torch.no_grad():
        torch.testing.assert_close(model(shifted, torch.zeros(1,4))-model(sequence, torch.zeros(1,4)), torch.full((1,3), 3.))


def test_future_row_cannot_enter_model_history():
    with pytest.raises(ValueError):
        TemporalLinear(1)(torch.zeros(1,169,2), torch.zeros(1,4))
