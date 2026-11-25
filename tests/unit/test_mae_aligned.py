"""Forward-only objective checks; no fitting, backward or optimizer steps."""

import pytest
import torch

from src.models.mae_aligned import MAEAlignedMLP
from src.models.neural_hurdle import NeuralHurdleMLP


def test_direct_projection_matches_exact_zero_scoring_loss():
    model = MAEAlignedMLP(target_mean=.5, target_scale=2.).eval()
    sequence = torch.zeros(2, 168, 8)
    sequence[:, :, :4] = -1
    phase = torch.zeros(2, 4)
    with torch.inference_mode():
        prediction = model(sequence, phase)
        physical = model.predict_physical(sequence, phase)
        loss, n = model.training_loss(sequence, phase, torch.full((2, 3), -.25))
    assert torch.all(prediction == -.25)
    assert torch.equal(physical, torch.zeros_like(physical))
    assert n == 6 and loss == 0


def test_mixture_objective_equals_point_mae_plus_fixed_surrogate():
    model = MAEAlignedMLP(mode="hurdle_mae", auxiliary_weight=.1).eval()
    old = NeuralHurdleMLP().eval()
    old.load_state_dict(model.state_dict())
    x, phase = torch.zeros(2, 168, 8), torch.zeros(2, 4)
    target = torch.tensor([[0., 1., float("nan")], [0., 2., 3.]])
    with torch.inference_mode():
        loss, n = model.training_loss(x, phase, target)
        surrogate, _ = old.training_loss(x, phase, target)
        mask = target.isfinite()
        point = (model(x, phase)[mask]-target[mask]).abs().mean()
        empty, count = model.training_loss(x, phase, torch.full((2, 3), float("nan")))
    torch.testing.assert_close(loss, point+.1*surrogate)
    assert n == 5 and empty == 0 and count == 0


def test_local_training_guard(monkeypatch):
    from scripts import train_mae_aligned

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(train_mae_aligned, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        train_mae_aligned.main()
