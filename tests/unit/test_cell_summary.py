"""Verifier fixtures and forward-only replay; no training or backward calls."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.summarize_cell_attention import predict_cells, verify_pretraining
from src.data.io import sha256
from src.data.panel import Panel
from src.features.cell_inputs import cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention
from src.models.cell_pretraining import SourcePatchForecaster, transfer_encoder
from src.utils.config import load_config


def test_replay_reads_only_past_and_preserves_group_axis():
    class Capture(torch.nn.Module):
        def forward(self, history, phase):
            self.history = history.clone()
            assert history.shape == (2, 168, 32, 8) and phase.shape == (2, 4)
            return torch.ones(2, 32, 3)

    panel = Panel(np.ones((504, 64, 4)), np.arange(504),
                  tuple(f"0_{c}_{b}" for c in range(2) for b in range(32)), {})
    preprocessing = GRUPreprocessor.fit(panel, 335, "B")
    model = Capture()
    original = predict_cells(model, panel, preprocessing, 383, cell_groups(panel.beam_ids))
    before = model.history
    changed = panel.values.copy()
    changed[384:] = 99999
    replay = predict_cells(model, replace(panel, values=changed), preprocessing, 383, cell_groups(panel.beam_ids))
    torch.testing.assert_close(before, model.history)
    np.testing.assert_array_equal(original, replay)
    with pytest.raises(ValueError, match="causal origin"):
        predict_cells(model, panel, preprocessing, 166, cell_groups(panel.beam_ids))


def test_transfer_verifier_rejects_source_head_and_wrong_budget(tmp_path):
    cfg = load_config("configs/cell_multisource.yaml")
    fold = cfg["folds"][0]
    source_root = tmp_path/"pretraining"/fold["name"]
    source_root.mkdir(parents=True)
    case = "target_self_pretrain"
    fixed = {**cfg, "batch_size": cfg["pretraining_batch_size"], "max_epochs": cfg["pretraining_epochs"],
             "early_stopping_patience": cfg["pretraining_epochs"]+1}
    (source_root/"source_meta.json").write_text(json.dumps({"config": fixed,
        "identity": {"case": case, "provenance": {"original_fit_end": fold["fit_end"],
                      "maximum_label_hour": fold["fit_end"]}}, "checkpoint": "last.pt"}))
    # These are deliberately constructed verifier fixtures, never a trained artifact.
    source = SourcePatchForecaster(cfg)
    last = {"epoch": 19, "model": source.state_dict(), "optimizer": {"state": {0: {"step": torch.tensor(5120)}}}}
    checkpoint = source_root/"last.pt"
    torch.save(last, checkpoint)
    torch.manual_seed(cfg["seed"])
    model = CellPatchAttention(cfg, 5, 1, "cell")
    transferred = transfer_encoder(model, source.state_dict())
    directory = tmp_path/fold["name"]/"cell"
    directory.mkdir(parents=True)
    initialization = {"model": model.state_dict(), "source_checkpoint_sha256": sha256(checkpoint),
                      "case": case, "fold": fold, "transferred_encoder_keys": transferred}
    torch.save(initialization, directory/"initialization.pt")
    verified = verify_pretraining(tmp_path, directory, model, fold, case, cfg)
    assert verified["optimizer_steps"] == 5120
    initialization["transferred_encoder_keys"] = transferred+["head.0.weight"]
    torch.save(initialization, directory/"initialization.pt")
    with pytest.raises(ValueError, match="Source head"):
        verify_pretraining(tmp_path, directory, model, fold, case, cfg)
    last["optimizer"]["state"][0]["step"] = torch.tensor(100)
    torch.save(last, checkpoint)
    with pytest.raises(ValueError, match="budget incomplete"):
        verify_pretraining(tmp_path, directory, model, fold, case, cfg)
