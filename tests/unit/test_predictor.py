"""Causality, request ordering, read-only data and NaN guards without fitting."""

import json

import numpy as np
import pytest

from src.data.panel import Panel
from src.inference.predictor import Predictor
from tests.utils.demo import baseline_assets


def test_readonly_history_order_and_no_future_input(tmp_path):
    p = Predictor(baseline_assets(tmp_path / "demo"))
    first = p.predict(["0_0_1", "0_0_0"], 199)
    assert first["beams"][0]["predictions"][0]["point"] == 0
    assert len(first["beams"][1]["history"]) == 168
    assert first["beams"][1]["history"][0]["time"] == 32
    assert not p.store.panel.values.flags.writeable
    history = p.store.history(["0_0_0"], 199)
    history[:] = -100
    assert p.store.panel.values[199, 0, 0] != -100
    values = p.store.panel.values.copy()
    values[200:] = 99999
    p.store.panel = Panel(values, p.store.panel.time, p.store.panel.beam_ids, {})
    second = p.predict(["0_0_1", "0_0_0"], 199)
    assert [b["history"] for b in first["beams"]] == [b["history"] for b in second["beams"]]
    assert [[v["point"] for v in b["predictions"]] for b in first["beams"]] == [
        [v["point"] for v in b["predictions"]] for b in second["beams"]
    ]
    assert second["beams"][0]["predictions"][0]["actual"] == 99999


def test_exclusion_nan_and_unknown_actual_are_json_safe(tmp_path):
    p = Predictor(baseline_assets(tmp_path / "demo"))
    risk = p.predict_risk(["0_0_1"], 239)
    assert all(
        r["status"] == "Excluded" and not r["valid_mask"] and r["actual"] is None
        for r in risk["beams"][0]["predictions"]
    )
    values = p.store.panel.values.copy()
    values[239, 0, 0] = np.nan
    p.store.panel = Panel(values, p.store.panel.time, p.store.panel.beam_ids, {})
    risk = p.predict_risk(["0_0_0"], 239)
    assert risk["beams"][0]["predictions"][0]["point"] is None
    assert risk["beams"][0]["predictions"][0]["reason"] == "missing_prediction"
    json.dumps(risk, allow_nan=False)


def test_corrupt_assets_refused(tmp_path):
    root = baseline_assets(tmp_path / "demo")
    (root / "replay.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        Predictor(root)


def test_all_missing_history_is_not_a_valid_alarm(tmp_path):
    p = Predictor(baseline_assets(tmp_path / "demo"))
    values = p.store.panel.values.copy()
    values[32:200, 0, 0] = np.nan
    p.store.panel = Panel(values, p.store.panel.time, p.store.panel.beam_ids, {})

    class Constant:
        def predict(self, history, horizon):
            return np.ones(
                history.shape[1]
            )  # A finite fallback prediction is still unsupported by observations.

    p.models = {h: Constant() for h in (1, 6, 24)}
    output = p.predict_risk(["0_0_0"], 199)
    assert all(
        v["status"] == "Excluded" and not v["valid_mask"] and v["reason"] == "missing_history"
        for v in output["beams"][0]["predictions"]
    )
