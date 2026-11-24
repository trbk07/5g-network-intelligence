"""Initialized inference checks only; no backward, optimizer or model fitting."""

import pytest
import torch

from src.models.temporal_tcn import CausalResidualBlock, TemporalResidualTCN


@pytest.mark.parametrize("kpis",[1,4])
def test_initialized_output_is_seasonal_and_batch_independent(kpis):
    model=TemporalResidualTCN(kpis).eval()
    sequence=torch.arange(168.)[None,:,None].repeat(2,1,2*kpis)
    phase=torch.zeros(2,4)
    with torch.inference_mode():
        result=model(sequence,phase)
        torch.testing.assert_close(result,torch.tensor([[72.,77.,95.],[72.,77.,95.]]))
        torch.testing.assert_close(result[:1],model(sequence[:1],phase[:1]))


def test_nonzero_block_cannot_read_later_time_steps():
    block=CausalResidualBlock(4,4,.1).eval()
    x=torch.randn(2,4,32)
    changed=x.clone()
    changed[:,:,12:]+=1000
    with torch.inference_mode():
        torch.testing.assert_close(block(x)[:,:,:12],block(changed)[:,:,:12])


def test_invalid_sequence_contract_fails():
    with pytest.raises(ValueError):
        TemporalResidualTCN(1)(torch.zeros(1,169,2),torch.zeros(1,4))


def test_shared_neural_trainer_rejects_local_before_loading_or_fitting(monkeypatch):
    from scripts.train_temporal_mlp import train_study

    monkeypatch.setattr("src.utils.runtime.is_cloud_runtime",lambda:False)
    with pytest.raises(RuntimeError,match="forbidden"):
        train_study(config_path="must-not-be-opened.yaml")
