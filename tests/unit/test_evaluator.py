import json

import numpy as np
import pytest

from src.data.panel import Panel
from src.evaluation.evaluator import evaluate, save_forecasts, to_benchmark_rows


class LinearOracle:
    name = "linear_oracle"

    def predict(self, history, horizon):
        assert history.shape[0] == 168
        assert not history.flags.writeable
        return history[-1, :, 0] + horizon


class Constant:
    name = "constant"

    def predict(self, history, horizon):
        return np.zeros(history.shape[1])


def fixture_panel():
    return Panel(np.broadcast_to(np.arange(200)[:, None, None], (200, 2, 4)).copy().astype(float),
                 np.arange(200), ("0_0_0", "0_0_1"), {})


def test_oracle_constant_and_replay_storage(tmp_path):
    panel = fixture_panel()
    origins = np.array([167, 168])
    record = evaluate(LinearOracle(), panel, origins, 6)
    assert to_benchmark_rows(record)[0]["mae"] == 0
    constant = evaluate(Constant(), panel, origins, 6)
    assert to_benchmark_rows(constant)[0]["mae"] == 173.5
    path = tmp_path / "forecasts.npz"
    save_forecasts([record, constant], path)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["record_0_predicted"], record.actual)
        np.testing.assert_array_equal(data["record_0_targets"], origins + 6)
        assert len(json.loads(str(data["metadata"]))["records"]) == 2


def test_missing_coverage_and_invalid_origins():
    panel = fixture_panel()
    panel.values[168, 0, 0] = np.nan
    row = to_benchmark_rows(evaluate(Constant(), panel, np.array([167]), 1))[0]
    assert row["n_missing_target"] == row["n_excluded"] == 1
    assert row["n_excluded_beams"] == 1
    for origins in (np.array([166]), np.array([199]), np.array([167, 167]), np.array([167.5])):
        with pytest.raises(ValueError):
            evaluate(Constant(), panel, origins, 1)
