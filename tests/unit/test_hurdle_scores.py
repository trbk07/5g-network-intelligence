"""Metric reconstruction checks with analytical arrays; no fitting."""

import numpy as np
import pytest

from scripts.summarize_hurdle_research import score_rows


def test_scores_use_all_observed_targets_and_zero_events():
    rows = score_rows(np.array([0, 2, np.nan, 4]),
                      {"candidate": np.array([1, 1, np.nan, 6])}, "inner1", 1)
    assert rows[0]["mae"] == pytest.approx(4 / 3)
    assert rows[0]["wape"] == pytest.approx(4 / 6)
    assert rows[0]["n_valid"] == 3
    assert rows[0]["zero_fp_rate"] == 1


@pytest.mark.parametrize("prediction", [[1, np.nan], [1]])
def test_missing_or_unaligned_prediction_cannot_shrink_population(prediction):
    with pytest.raises(ValueError):
        score_rows(np.array([1, 2]), {"candidate": prediction}, "inner1", 6)


def test_zero_denominator_does_not_fabricate_percentage():
    with pytest.raises(ValueError):
        score_rows(np.array([0, 0]), {"candidate": [0, 0]}, "inner1", 24)
