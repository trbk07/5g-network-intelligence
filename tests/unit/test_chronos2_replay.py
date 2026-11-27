"""Mock inference only; no external model instantiation or training."""

import numpy as np
import pytest

from src.evaluation.chronos2_replay import cached_origin, predict_origin


class Pipeline:
    quantiles = [.1, .5, .9]

    def __init__(self):
        self.calls = 0

    def predict(self, tasks, **kwargs):
        self.calls += 1
        assert kwargs["cross_learning"] is False
        assert kwargs["context_length"] == 168
        outputs = []
        for task in tasks:
            assert set(task) == {"target", "past_covariates"}
            assert len(task["past_covariates"]) == 3
            point = task["target"][-1]+np.arange(1, 25)
            outputs.append(np.stack([point-1, point, point+1])[None])
        return outputs


def test_mock_replay_is_causal_and_covers_every_beam():
    panel = np.zeros((840, 3, 4))
    panel[383, :, 0] = [1, 2, 3]
    panel[384:] = np.inf
    result = predict_origin(Pipeline(), panel, 383)
    assert result.shape == (3, 24)
    np.testing.assert_array_equal(result[:, 0], [2, 3, 4])
    np.testing.assert_array_equal(result[:, -1], [25, 26, 27])


def test_cache_reuses_only_same_model_and_history(tmp_path):
    panel = np.zeros((840, 3, 4))
    pipeline = Pipeline()
    first = cached_origin(pipeline, panel, 383, tmp_path, {"base": "pinned", "adapter": "one"})
    np.testing.assert_array_equal(first, cached_origin(pipeline, panel, 383, tmp_path,
                                                     {"base": "pinned", "adapter": "one"}))
    assert pipeline.calls == 1
    with pytest.raises(ValueError, match="different model/history"):
        cached_origin(pipeline, panel, 383, tmp_path, {"base": "pinned", "adapter": "two"})
    panel[383, 0, 0] = 1
    with pytest.raises(ValueError, match="different model/history"):
        cached_origin(pipeline, panel, 383, tmp_path, {"base": "pinned", "adapter": "one"})


def test_stopping_does_not_read_scoring_labels_and_scoring_keeps_full_axes(tmp_path):
    from types import SimpleNamespace

    from scripts.evaluate_chronos2_lora import stopping_mae, score_case

    values = np.broadcast_to(np.arange(504.)[:, None, None], (504, 3, 4)).copy()
    values[:, 2, 0] = 0
    fold = {"name": "inner1", "fit_end": 335, "stop_end": 383, "score_end": 443}
    stopping = values.copy()
    stopping[384:] = np.inf
    result = stopping_mae(Pipeline(), stopping, fold, tmp_path/"stop", {"model": "fixture"}, 256)
    assert result["n_valid"] == 25*3*3
    assert result["mae"] == pytest.approx(31/9)
    values[444:] = np.inf
    panel = SimpleNamespace(values=values, beam_ids=("one", "two", "three"))
    rows = score_case(Pipeline(), panel, fold, "fixture", tmp_path/"score", {"model": "fixture"}, 256)
    assert [r["n_valid"] for r in rows] == [180, 165, 111]
    assert [r["mae"] for r in rows] == pytest.approx([1/3, 2, 8])


def test_real_checkpoint_selection_refuses_local_before_external_model_import(monkeypatch):
    from scripts import evaluate_chronos2_lora

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(evaluate_chronos2_lora, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        evaluate_chronos2_lora.main()
