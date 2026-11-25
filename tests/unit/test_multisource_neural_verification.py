"""Verification failures on static score tables/checkpoints; no model training."""

import pandas as pd
import pytest
import torch

from scripts.summarize_multisource_neural import compare_cases, verify_initialization
from src.data.io import sha256


def frames():
    rows = [{"fold": fold, "variant": "A", "horizon": h, "mae": 1., "wape": .5,
             "n_valid": 100, "zero_fp_rate": .9}
            for fold in ("inner1", "inner2") for h in (1, 6, 24)]
    return {case: pd.DataFrame(rows) for case in
            ("target_only", "target_self_pretrain", "radio", "cesnet", "radio_and_cesnet")}


def test_comparison_has_both_controls():
    data = frames()
    data["radio"]["mae"] = .8
    result = compare_cases(data)
    assert len(result) == 30
    assert result.loc[result.case == "radio", "relative_mae_gain_vs_target_self_pretrain_pct"].tolist() == pytest.approx([20.]*6)


def test_missing_self_control_rejected():
    data = frames()
    del data["target_self_pretrain"]
    with pytest.raises(ValueError, match="five cases"):
        compare_cases(data)


def test_shrunk_coverage_rejected():
    data = frames()
    data["cesnet"].loc[0, "n_valid"] = 99
    with pytest.raises(AssertionError):
        compare_cases(data)


def test_transferred_weights_must_equal_source(tmp_path):
    fold = {"name": "inner1", "fit_end": 335}
    source = tmp_path/"external/radio/last.pt"
    source.parent.mkdir(parents=True)
    torch.save({"epoch": 4, "model": {"weight": torch.tensor([1.])}}, source)
    directory = tmp_path/"radio/inner1/A"
    directory.mkdir(parents=True)
    initial = {"case": "radio", "fold": fold, "source_checkpoint": str(source),
               "source_checkpoint_sha256": sha256(source), "model": {"weight": torch.tensor([1.])}}
    torch.save(initial, directory/"initialization.pt")
    verify_initialization(tmp_path, "radio", fold, {"external_epochs": 5})
    initial["model"]["weight"] = torch.tensor([2.])
    torch.save(initial, directory/"initialization.pt")
    with pytest.raises(AssertionError):
        verify_initialization(tmp_path, "radio", fold, {"external_epochs": 5})
