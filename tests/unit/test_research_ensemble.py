"""Inference and chronological boundaries without fitting or local weight calibration."""

import json

import numpy as np
import pytest

from scripts.finalize_improvement import choose_weights
from scripts.optimize_models import fold_arrays
from src.data.panel import Panel
from src.models.research_ensemble import ResearchEnsemble


def test_baseline_ensemble_history_only_and_clip(tmp_path):
    members = {
        "persistence": {"family": "baseline", "path": None, "weight": 0.6},
        "seasonal24": {"family": "baseline", "path": None, "weight": 0.4},
    }
    plan = {
        "schema": "causal40-ensemble-research-v1",
        "fit_end": 503,
        "routes": {str(h): {"members": members, "bias": -0.01} for h in (1, 6, 24)},
    }
    (tmp_path / "ensemble.json").write_text(json.dumps(plan))
    model = ResearchEnsemble(tmp_path)
    history = np.zeros((168, 2, 4))
    history[:, :, 0] = np.arange(168)[:, None]
    np.testing.assert_allclose(model.predict_at(history, 503, 6), 0.6 * 167 + 0.4 * 149 - 0.01)
    assert np.array_equal(model.predict_at(np.zeros_like(history), 503, 6), [0, 0])
    with pytest.raises(ValueError, match="Invalid research"):
        model.predict_at(history, 502, 6)


def test_folds_have_distinct_target_windows():
    panel = Panel(np.zeros((504, 2, 4)), np.arange(504), ("0_0_0", "0_0_1"), {})
    fold = {"fit_end": 335, "stop_end": 383, "score_end": 443}
    for h in (1, 6, 24):
        fit, stop, score, *_ = fold_arrays(panel, fold, h, 2)
        assert (fit + h).max() <= 335
        assert (stop + h).min() == 335 + h and (stop + h).max() == 383
        assert (score + h).min() == 383 + h and (score + h).max() == 443


def test_no_local_weight_training():
    from src.utils.runtime import is_cloud_runtime

    if is_cloud_runtime():
        pytest.skip("Local-denial test")
    with pytest.raises(RuntimeError, match="Local model training is forbidden"):
        choose_weights(None, None, {})
