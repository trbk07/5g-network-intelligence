"""Initialized forward-only checks; no model fitting, backward or optimizer."""

import pytest
import torch

from src.models.temporal_mlp import TemporalResidualMLP


@pytest.mark.parametrize("kpis",[1,4])
def test_initial_output_equals_seasonal_average_and_is_batch_independent(kpis):
    model=TemporalResidualMLP(kpis).eval()
    sequence=torch.arange(168.)[None,:,None].repeat(2,1,2*kpis)
    phase=torch.zeros(2,4)
    with torch.no_grad():
        output=model(sequence,phase)
        torch.testing.assert_close(output,torch.tensor([[72.,77.,95.],[72.,77.,95.]]))
        torch.testing.assert_close(output[:1],model(sequence[:1],phase[:1]))


def test_history_shape_and_phase_contract():
    with pytest.raises(ValueError):
        TemporalResidualMLP(1)(torch.zeros(1,169,2),torch.zeros(1,4))
    with pytest.raises(ValueError):
        TemporalResidualMLP(4)(torch.zeros(1,168,8),torch.zeros(1,5))
