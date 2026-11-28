"""Loss arithmetic, prefix counts and guards only; no local training."""

import numpy as np
import pytest
import torch

from scripts.train_partial_supervision import supervision_counts
from src.data.panel import Panel
from src.models.partial_supervision import horizon_balanced_l1
from scripts.configure_partial_supervision import protocols
from src.utils.config import load_config


def test_horizons_have_equal_weight_despite_different_label_counts():
    actual = torch.tensor([[[1., 2., 3.], [3., float("nan"), 7.]]])
    loss, count = horizon_balanced_l1(torch.zeros_like(actual), actual)
    assert float(loss) == 3 and count == 5  # means 2, 2 and 5, then average.


def test_missing_horizon_and_all_missing_are_explicit():
    actual = torch.tensor([[[1., float("nan"), 3.], [3., float("nan"), 7.]]])
    loss, count = horizon_balanced_l1(torch.zeros_like(actual), actual)
    assert float(loss) == 3.5 and count == 4
    loss, count = horizon_balanced_l1(torch.ones_like(actual), torch.full_like(actual, float("nan")))
    assert float(loss) == 0 and count == 0


def test_nonfinite_predictions_cannot_shrink_supervision():
    with pytest.raises(ValueError, match="Finite predictions"):
        horizon_balanced_l1(torch.full((1, 2, 3), float("nan")), torch.ones(1, 2, 3))


def test_supervision_counts_ignore_poisoned_future():
    values = np.ones((504, 32, 4))
    values[336:] = np.nan
    panel = Panel(values, np.arange(504), tuple(f"0_0_{b}" for b in range(32)), {})
    counts = supervision_counts(panel, np.arange(167, 335), 335)
    assert counts == {"1": 168*32, "6": 163*32, "24": 145*32}


def test_controller_refuses_local_training(monkeypatch):
    from scripts import train_partial_supervision

    def reject():
        raise RuntimeError("local training forbidden")

    monkeypatch.setattr(train_partial_supervision, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local training forbidden"):
        train_partial_supervision.main()


def test_new_joint_control_uses_same_loss_and_target_protocol():
    base = load_config("configs/cell_attention_v2.yaml")
    arms = protocols(base)
    for key in ("seed", "folds", "width", "learning_rate", "max_epochs", "early_stopping_patience"):
        assert arms["joint"][key] == arms["available"][key] == base[key]
    assert arms["joint"]["training_objective"] == arms["available"]["training_objective"]
    assert arms["joint"]["stopping_label_policy"] == arms["available"]["stopping_label_policy"]
    assert base["modes"] == ["local", "cell"]
