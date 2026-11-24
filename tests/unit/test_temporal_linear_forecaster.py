"""Initialized checkpoint serialization/inference checks only; no training."""

import json

import numpy as np
import pytest
import torch

from src.data.io import sha256
from src.features.gru_inputs import GRUPreprocessor
from src.models.temporal_linear import TemporalLinear
from src.models.temporal_linear_forecaster import TemporalLinearForecaster
from src.models.temporal_mlp import TemporalResidualMLP
from src.models.temporal_mlp_forecaster import TemporalMLPForecaster
from src.models.temporal_tcn import TemporalResidualTCN
from src.models.temporal_tcn_forecaster import TemporalTCNForecaster


def fixture_checkpoint(root, score_end=443, kind="linear"):
    folder = root / "inner1/A"
    folder.mkdir(parents=True)
    fold = {"name": "inner1", "fit_end": 335, "stop_end": 383, "score_end": score_end}
    meta = {"fold": fold, "variant": "A", "holdout_used": False, "validation_504_671_used": False,
            "protocol": {"protocol": "temporal-linear-development-v1", "seed": 17}, "training": {"best_epoch": 0}}
    if kind=="mlp":
        meta["protocol"].update(protocol="temporal-residual-mlp-development-v1", hidden_size=64, dropout=0.1)
    elif kind=="tcn":
        meta["protocol"].update(protocol="temporal-residual-tcn-development-v1", channels=32, levels=7, dropout=.1)
    (folder / "meta.json").write_text(json.dumps(meta))
    # Fixed constants and initialized state: no fitted preprocessing or model.
    GRUPreprocessor(np.zeros(1), np.zeros(1), np.ones(1), ("0_0_0", "0_0_1"), 335, "A").save(folder / "preprocessor.npz")
    model = {"linear":TemporalLinear,"mlp":TemporalResidualMLP,"tcn":TemporalResidualTCN}[kind](1)
    architecture = f"temporal-residual-{kind}-v1" if kind!="linear" else "temporal-linear-v1"
    torch.save({"model": model.state_dict(), "best_epoch": 0,
                "identity": {"seed": 17, "run_identity": {"architecture": architecture, "fold": fold, "variant": "A"}}}, folder / "best.pt")
    files = {p.relative_to(root).as_posix(): sha256(p) for p in folder.iterdir()}
    (root / "manifest.json").write_text(json.dumps({"files": files}))


@pytest.mark.parametrize("kind", ["linear","mlp","tcn"])
def test_initialized_roundtrip_recovers_causal_seasonal_mean(tmp_path, kind):
    fixture_checkpoint(tmp_path, kind=kind)
    loader = {"linear":TemporalLinearForecaster,"mlp":TemporalMLPForecaster,"tcn":TemporalTCNForecaster}[kind]
    model = loader(tmp_path, "inner1", "A")
    history = np.zeros((168, 2, 4))
    history[:, :, 0] = np.arange(168)[:, None] + np.array([0, 2])
    for h in (1, 6, 24):
        np.testing.assert_allclose(model.predict_at(history, 400, h), [71+h, 73+h], atol=1e-5)


def test_corrupted_checkpoint_is_rejected_before_loading(tmp_path):
    fixture_checkpoint(tmp_path)
    (tmp_path / "inner1/A/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        TemporalLinearForecaster(tmp_path, "inner1", "A")


def test_metadata_cannot_expand_development_into_validation(tmp_path):
    fixture_checkpoint(tmp_path, score_end=504)
    with pytest.raises(ValueError, match="temporal contract"):
        TemporalLinearForecaster(tmp_path, "inner1", "A")
