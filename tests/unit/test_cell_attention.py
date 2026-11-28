"""Forward-only topology/causality tests; no fitting, backward or optimizer steps."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.data.panel import Panel
from src.features.cell_inputs import CellBatches, cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention, configure_attention_math
from src.utils.config import load_config
from src.utils.seed import set_neural_seed


def panel_fixture():
    values = np.broadcast_to(np.arange(504.)[:, None, None], (504, 32, 4)).copy()
    return Panel(values, np.arange(504), tuple(f"0_0_{b}" for b in range(32)), {})


def test_topology_and_future_poisoning():
    panel = panel_fixture()
    assert cell_groups(panel.beam_ids).shape == (1, 32)
    with pytest.raises(ValueError, match="32"):
        cell_groups(panel.beam_ids[:-1])
    pre = GRUPreprocessor.fit(panel, 335, "B")
    a = next(CellBatches(panel, pre, [383]).batches(1))
    poisoned = panel.values.copy()
    poisoned[384:] += 10000
    b = next(CellBatches(replace(panel, values=poisoned), pre, [383]).batches(1))
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    assert a[0].shape == (1, 168, 32, 8) and a[2].shape == (1, 32, 3)
    assert not torch.equal(a[2], b[2])


def test_local_ablation_blocks_neighbours_and_matches_parameter_count():
    cfg = load_config("configs/cell_attention.yaml")
    torch.manual_seed(17)
    local = CellPatchAttention(cfg, 5, 1, "local").eval()
    cell = CellPatchAttention(cfg, 5, 1, "cell").eval()
    cell.load_state_dict(local.state_dict())
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in cell.parameters())
    # Fixed analytic head activation allows testing dependencies without fitting.
    with torch.no_grad():
        local.head[-1].weight.fill_(.05)
        cell.load_state_dict(local.state_dict())
        history = torch.zeros(1, 168, 32, 8)
        changed = history.clone()
        changed[:, 30:80, 1, :4] = 10
        phase = torch.zeros(1, 4)
        torch.testing.assert_close(local(history, phase)[:, 0], local(changed, phase)[:, 0])
        difference = (cell(history, phase)[:, 0]-cell(changed, phase)[:, 0]).abs().max()
        assert float(difference) > 1e-7
        assert torch.isfinite(cell(changed, phase)).all()


def test_entry_point_refuses_local_training(monkeypatch):
    from scripts import train_cell_attention

    def reject():
        raise RuntimeError("local training forbidden")

    monkeypatch.setattr(train_cell_attention, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local training forbidden"):
        train_cell_attention.main()


def test_explicit_attention_math_policy():
    original = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        configure_attention_math()
        assert not torch.backends.mha.get_fastpath_enabled()
    finally:
        torch.backends.mha.set_fastpath_enabled(original)


def test_neural_seed_matches_cold_weights_across_spatial_modes():
    cfg = load_config("configs/cell_attention_v2.yaml")
    set_neural_seed(cfg["seed"])
    local = CellPatchAttention(cfg, 1, 2, "local")
    # An unrelated initialization consumes RNG; reseeding must restore the cold start.
    torch.rand(100)
    set_neural_seed(cfg["seed"])
    spatial = CellPatchAttention(cfg, 1, 2, "cell")
    assert torch.initial_seed() == cfg["seed"]
    assert all(torch.equal(v, spatial.state_dict()[k]) for k, v in local.state_dict().items())
