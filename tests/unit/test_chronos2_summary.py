"""Independent metric/cache verification uses arrays only, without model execution."""

import json

import numpy as np
import pytest

from scripts.summarize_chronos2_lora import cached_forecast, metrics, verify_tree
from src.evaluation.chronos2_replay import cached_origin


def test_metrics_preserve_zeros_and_observation_mask():
    result = metrics([0., 2., np.nan, 2.], [1., 1., 50., 2.])
    assert result == {"mae": 2/3, "wape": .5, "n_valid": 3, "zero_fp_rate": 1.}
    with pytest.raises(ValueError, match="coverage"):
        metrics([0., 2.], [0., np.nan])
    with pytest.raises(ValueError, match="denominator"):
        metrics([0., 0.], [0., 1.])


def test_cached_forecast_checks_causal_identity_without_reinference(tmp_path):
    class Mock:
        quantiles = [.5]

        def predict(self, tasks, **kwargs):
            return [np.ones((1, 1, 24)) for _ in tasks]

    values = np.ones((504, 3, 4))
    cached_origin(Mock(), values, 383, tmp_path, {"model": "fixture"})
    values[384:] = 1000
    predicted = cached_forecast(tmp_path/"origin_383.npz", values, 383, {"model": "fixture"})
    assert predicted.shape == (3, 24)
    values[383, 0, 0] = 2
    with pytest.raises(ValueError, match="identity"):
        cached_forecast(tmp_path/"origin_383.npz", values, 383, {"model": "fixture"})


def test_tree_rejects_undeclared_or_unsafe_members(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"files": {}}))
    verify_tree(tmp_path)
    (tmp_path/"untracked.txt").write_text("fixture")
    with pytest.raises(ValueError, match="Unmanifested"):
        verify_tree(tmp_path)
    (tmp_path/"manifest.json").write_text(json.dumps({"files": {"../escape": "a"*64}}))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_tree(tmp_path)
