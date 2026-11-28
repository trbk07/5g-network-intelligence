"""Forward and causal-ablation tests only; no fitting/backward/optimizer steps."""

import pytest
import torch

from src.models.past_covariate_attention import PastCovariateAttention
from src.utils.config import load_config
from src.utils.seed import set_neural_seed


def test_cold_states_and_parameter_counts_are_matched():
    cfg = load_config("configs/past_covariate_attention.yaml")
    set_neural_seed(17)
    a = PastCovariateAttention(cfg, False).eval()
    set_neural_seed(17)
    b = PastCovariateAttention(cfg, True).eval()
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    history = torch.rand(2, 168, 8)
    with torch.no_grad():
        expected = history[:, a.seasonal_indices, 0].mean(-1)
        torch.testing.assert_close(a(history, torch.zeros(2, 4)), expected)
        torch.testing.assert_close(b(history, torch.zeros(2, 4)), expected)


def test_no_covariate_control_is_invariant_but_bridge_can_use_observed_covariates():
    cfg = load_config("configs/past_covariate_attention.yaml")
    set_neural_seed(17)
    a = PastCovariateAttention(cfg, False).eval()
    b = PastCovariateAttention(cfg, True).eval()
    # Activate a deterministic diagnostic head without any training.
    with torch.no_grad():
        a.head.weight.copy_(torch.randn_like(a.head.weight)*.005)
        b.load_state_dict(a.state_dict())
        history = torch.ones(2, 168, 8)
        history[:, :, 4:] = 0
        changed = history.clone()
        changed[:, 20:100, 1:4] = 20
        phase = torch.zeros(2, 4)
        assert torch.equal(a(history, phase), a(changed, phase))
        assert float((b(history, phase)-b(changed, phase)).abs().max()) > 1e-7


def test_scaling_and_input_boundary_are_explicit():
    model = PastCovariateAttention(load_config("configs/past_covariate_attention.yaml")).eval()
    model.configure_target_scale(.5, .25)
    with torch.no_grad():
        physical = model.predict_physical(torch.zeros(1, 168, 8), torch.zeros(1, 4))
    torch.testing.assert_close(physical, torch.full((1, 3), .5))
    with pytest.raises(ValueError, match="168 observed"):
        model(torch.zeros(1, 169, 8), torch.zeros(1, 4))
    with pytest.raises(ValueError, match="positive scale"):
        model.configure_target_scale(0, 0)


def test_controller_refuses_local_training(monkeypatch):
    from scripts import train_past_covariate_attention

    def reject():
        raise RuntimeError("local training forbidden")

    monkeypatch.setattr(train_past_covariate_attention, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local training forbidden"):
        train_past_covariate_attention.main()
