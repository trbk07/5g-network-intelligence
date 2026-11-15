import numpy as np
import pytest

from src.models.baselines import Persistence, SeasonalNaive


@pytest.mark.parametrize("h", [1, 6, 24])
def test_causal_reference_indices(h):
    history = np.broadcast_to(np.arange(168)[:, None, None], (168, 2, 4))
    np.testing.assert_array_equal(Persistence().predict(history, h), [167, 167])
    for period in (24, 168):
        index = 167 + h - period
        assert 0 <= index <= 167
        np.testing.assert_array_equal(SeasonalNaive(period).predict(history, h), [index, index])
    if h == 24:
        np.testing.assert_array_equal(Persistence().predict(history, h),
                                      SeasonalNaive(24).predict(history, h))


def test_reject_negative_and_future_reference_and_keep_nan():
    with pytest.raises(ValueError):
        SeasonalNaive(168).predict(np.zeros((2, 1, 4)), 1)
    with pytest.raises(ValueError):
        SeasonalNaive(24).predict(np.zeros((168, 1, 4)), 25)
    history = np.zeros((168, 1, 4))
    history[-1, 0, 0] = np.nan
    assert np.isnan(Persistence().predict(history, 1)[0])
