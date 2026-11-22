"""Fixed-coefficient arithmetic only; no weight search or calibration."""

import numpy as np
import pytest

from src.models.forecast_blend import blend_forecasts


def test_known_convex_combination_preserves_shape_and_precision():
    hurdle, linear = np.array([0, 2], dtype=np.float32), np.array([4, 0], dtype=np.float32)
    np.testing.assert_allclose(blend_forecasts(hurdle, linear, 0.25), [1, 1.5])
    np.testing.assert_array_equal(blend_forecasts(hurdle, linear, 0), hurdle)
    assert blend_forecasts(hurdle, linear, 0.25).dtype == np.float64


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan])
def test_bad_weight_fails(weight):
    with pytest.raises(ValueError):
        blend_forecasts([1], [2], weight)


def test_misaligned_or_missing_forecasts_cannot_change_coverage():
    with pytest.raises(ValueError):
        blend_forecasts([1, 2], [2], 0.1)
    with pytest.raises(ValueError):
        blend_forecasts([1], [np.nan], 0.1)
