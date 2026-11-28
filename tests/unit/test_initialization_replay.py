"""Initialization arithmetic only; no fitting or optimizer operations."""

import pytest
import torch

from src.models.initialization_replay import hosted_cold_model, verify_cold_state


def test_hosted_linux_seed17_fixture_and_rng_restoration():
    before = torch.get_rng_state().clone()
    model = hosted_cold_model(lambda: torch.nn.Linear(96, 64), 17)
    # First eight float32 weights from archived Kaggle seed-17 cold checkpoint.
    expected = torch.tensor([-0.01342295203357935, 0.007166713010519743, 0.06740357726812363,
                             -0.07677853107452393, -0.09607693552970886, 0.010082936845719814,
                             -0.02398640289902687, 0.009442722424864769])
    assert torch.equal(model.weight.flatten()[:8], expected)
    assert torch.equal(before, torch.get_rng_state())
    assert verify_cold_state(lambda: torch.nn.Linear(96, 64), 17, model.state_dict()) in ("native", "hosted_fma")
    with pytest.raises(ValueError, match="exactly reproduce"):
        verify_cold_state(lambda: torch.nn.Linear(96, 64), 29, model.state_dict())
    changed = {k: v.clone() for k, v in model.state_dict().items()}
    changed["weight"][0, 0] += 1e-6
    with pytest.raises(ValueError, match="exactly reproduce"):
        verify_cold_state(lambda: torch.nn.Linear(96, 64), 17, changed)
