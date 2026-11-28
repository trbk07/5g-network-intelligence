"""Forward and causal-ablation tests only; no fitting/backward/optimizer steps."""

import pytest
import numpy as np
import torch

from dataclasses import replace

from src.data.panel import Panel
from src.features.gru_inputs import GRUBatches, GRUPreprocessor

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


def test_batch_forecast_does_not_depend_on_values_after_origin():
    """Poison every future KPI, not just the target, at the real batch interface."""
    rng = np.random.default_rng(17)
    values = rng.uniform(.1, 2., (504, 2, 4))
    panel = Panel(values, np.arange(504), ('0_0_0', '0_0_1'), {})
    pre = GRUPreprocessor.fit(panel, 335, 'B')  # Preprocessing only; no model fit.
    poisoned = values.copy()
    poisoned[384:] = 1e9
    original = next(GRUBatches(panel, pre, 'B', [383], [0, 1]).batches(2))
    altered = next(GRUBatches(replace(panel, values=poisoned), pre, 'B', [383], [0, 1]).batches(2))
    assert torch.equal(original[0], altered[0]) and torch.equal(original[1], altered[1])
    assert not torch.equal(original[2], altered[2])
    cfg = load_config('configs/past_covariate_attention.yaml')
    model = PastCovariateAttention(cfg).configure_target_scale(pre.mean[0], pre.scale[0]).eval()
    with torch.no_grad():
        # Activate the attention route analytically, without optimization.
        model.head.weight.copy_(torch.randn_like(model.head.weight)*.005)
        a = model.predict_physical(original[0], original[1])
        b = model.predict_physical(altered[0], altered[1])
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_standardized_loss_preserves_physical_mae_and_nonnegative_outputs():
    model = PastCovariateAttention(load_config('configs/past_covariate_attention.yaml')).eval()
    mean, scale = .7, .3
    model.configure_target_scale(mean, scale)
    sequence = torch.full((2, 168, 8), -10.)
    sequence[:, :, 4:] = 0
    phase = torch.zeros(2, 4)
    actual = torch.tensor([[0., .2, .5], [.1, .3, .8]])
    with torch.no_grad():
        standardized = model(sequence, phase)
        physical = model.predict_physical(sequence, phase)
        scaled_loss = torch.abs(standardized-(actual-mean)/scale).mean()
        physical_loss = torch.abs(physical-actual).mean()
    assert bool((physical >= 0).all())
    torch.testing.assert_close(scaled_loss*scale, physical_loss)
