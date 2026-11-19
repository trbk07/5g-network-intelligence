import numpy as np
import pytest

from src.evaluation.common_mask import comparable_mask
from src.evaluation.bootstrap import paired_cluster_ci


def test_common_pairs_and_cluster_oracle():
    actual = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])
    a = actual.copy()
    b = actual + 1
    b[0, 1] = np.nan
    mask = comparable_mask(actual, {"oracle": a, "bad": b})
    assert mask.tolist() == [[True, False, True], [True, True, False]]
    ci = paired_cluster_ci(actual, a, b, np.array([0, 1, 2]), mask, repetitions=100)
    assert ci["difference"] == ci["lower"] == ci["upper"] == -1
    with pytest.raises(ValueError):
        comparable_mask(actual, {"bad": np.zeros((2, 2))})


def test_bootstrap_identical_predictions_and_missing_cluster():
    actual = np.arange(12, dtype=float).reshape(3, 4)
    ci = paired_cluster_ci(actual, actual + 2, actual + 2, np.array([0, 0, 1, 1]), repetitions=100)
    assert ci["lower"] == ci["upper"] == 0
    missing = paired_cluster_ci(actual, actual, actual, np.zeros(4), repetitions=100)
    assert missing["status"] == "insufficient_clusters"
