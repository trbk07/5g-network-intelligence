"""Initialized inference and numerical-loss checks; no fitting/backward/optimizer."""

import numpy as np
import pytest
import torch

from src.models.hurdle import mixture_median
from src.models.neural_hurdle import NeuralHurdleMLP, conditional_mixture_median


def test_tensor_mixture_matches_existing_numpy_rule():
    p = torch.tensor([[0., .5, .51], [.7, .9, 1.]])
    quantiles = torch.tensor([.1, .2, .4, 1.]).expand(2, 3, 4)
    levels = torch.tensor([.05, .1, .25, .5])
    with torch.inference_mode():
        actual = conditional_mixture_median(p, quantiles, levels).numpy()
    expected = mixture_median(p.numpy().reshape(-1), quantiles.numpy().reshape(-1, 4),
                              np.array([.05, .1, .25, .5])).reshape(2, 3)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert np.all(actual[0, :2] == 0)


def test_initial_direct_matches_causal_baseline_and_hurdle_has_exact_zero():
    x = torch.arange(168.)[None, :, None].repeat(2, 1, 8)
    phase = torch.zeros(2, 4)
    with torch.inference_mode():
        direct = NeuralHurdleMLP(kind="direct").eval()(x, phase)
        hurdle = NeuralHurdleMLP(target_mean=.5, target_scale=2.).eval()(x, phase)
    torch.testing.assert_close(direct, torch.tensor([[72., 77., 95.]]).expand(2, -1))
    torch.testing.assert_close(hurdle*2+.5, torch.zeros(2, 3))
    with torch.inference_mode():
        physical = NeuralHurdleMLP(target_mean=.123456789, target_scale=.987654321).eval().predict_physical(x, phase)
    assert torch.equal(physical, torch.zeros_like(physical))


@pytest.mark.parametrize("kind", ["direct", "hurdle"])
def test_loss_masks_missing_and_accepts_all_zero_targets(kind):
    model = NeuralHurdleMLP(target_mean=.5, target_scale=2., kind=kind).eval()
    x, phase = torch.zeros(2, 168, 8), torch.zeros(2, 4)
    targets = torch.tensor([[-.25, float("nan"), .75], [-.25, -.25, -.25]])
    with torch.inference_mode():
        loss, n = model.training_loss(x, phase, targets)
        empty, n_empty = model.training_loss(x, phase, torch.full((2, 3), float("nan")))
        zeros, n_zero = model.training_loss(x, phase, torch.full((2, 3), -.25))
    assert n == 5 and torch.isfinite(loss) and loss >= 0
    assert n_empty == 0 and empty == 0
    assert n_zero == 6 and torch.isfinite(zeros)


def test_positive_quantiles_are_noncrossing_and_batch_independent():
    model = NeuralHurdleMLP().eval()
    x, phase = torch.zeros(2, 168, 8), torch.zeros(2, 4)
    with torch.inference_mode():
        _, quantiles = model.components(x, phase)
        assert torch.all(torch.diff(quantiles, dim=-1) > 0)
        torch.testing.assert_close(model(x, phase)[:1], model(x[:1], phase[:1]))


def test_training_entrypoint_rejects_local(monkeypatch):
    from scripts import train_neural_hurdle

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(train_neural_hurdle, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        train_neural_hurdle.main()


def test_history_adapter_preserves_exact_physical_zero():
    from scripts.summarize_neural_hurdle import HurdleForecaster
    from src.features.gru_inputs import GRUPreprocessor

    adapter = HurdleForecaster.__new__(HurdleForecaster)
    means = np.array([.123456789, 0., 0., 0.])
    scales = np.array([.987654321, 1., 1., 1.])
    adapter.preprocessor = GRUPreprocessor(np.zeros(4), means, scales, ("one", "two"), 335, "B")
    adapter.model = NeuralHurdleMLP(target_mean=means[0], target_scale=scales[0]).eval()
    result = adapter.predict_at(np.zeros((168, 2, 4)), 383, 6)
    assert np.array_equal(result, np.zeros(2))
    with pytest.raises(ValueError, match="168 historical"):
        adapter.predict_at(np.zeros((169, 2, 4)), 383, 6)
