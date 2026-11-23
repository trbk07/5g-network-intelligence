"""Analytical fixed-baseline checks; no training, optimizer or calibration."""

import numpy as np
import pytest

from src.models.seasonal_median import SeasonalMedian


def test_uses_exactly_seven_causal_phase_matches():
    history = np.zeros((168,2,4))
    history[:,:,0] = np.arange(168)[:,None]
    for h in (1,6,24):
        np.testing.assert_array_equal(SeasonalMedian().predict(history,h),[71+h,71+h])


def test_single_outlier_and_missing_day_do_not_change_typical_level():
    history = np.ones((168,2,4))
    history[0,0,0] = 10000
    history[24,0,0] = np.nan
    history[:,1,0] = np.nan
    prediction = SeasonalMedian().predict(history,1)
    assert prediction[0] == 1
    assert np.isnan(prediction[1])


@pytest.mark.parametrize("h", [0, True, 25])
def test_unsupported_horizons_fail(h):
    with pytest.raises(ValueError):
        SeasonalMedian().predict(np.zeros((168,1,4)),h)
