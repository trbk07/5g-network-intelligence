"""Analytical probability metric checks; no calibration fitting."""

import numpy as np
import pytest

from scripts.summarize_hurdle_calibration import probability_scores


def test_known_binary_losses_and_missing_target():
    scores = probability_scores([0, 2, np.nan], [0.25, 0.75, 0.5])
    assert scores["brier"] == pytest.approx(0.0625)
    assert scores["log_loss"] == pytest.approx(-np.log(0.75))


@pytest.mark.parametrize("actual,p", [([0, 1], [0.5]), ([1], [np.nan]), ([0], [1.1]), ([np.nan], [0.5])])
def test_invalid_probability_population_fails(actual, p):
    with pytest.raises(ValueError):
        probability_scores(actual, p)
