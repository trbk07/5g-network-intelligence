"""Analytical inference checks; no model training."""

import numpy as np
import pytest

from src.models.hurdle import mixture_median


def test_mixture_uniform_positive_has_analytical_median():
    # Conditional positive distribution Uniform(0, 10).
    levels = np.array([0.05, 0.1, 0.25, 0.5])
    p = np.array([0, 0.4, 0.5, 0.6, 0.75, 1])
    values = np.tile(levels * 10, (len(p), 1))
    expected = np.where(p > 0.5, 10 * (1 - 0.5 / np.maximum(p, 0.01)), 0)
    np.testing.assert_allclose(mixture_median(p, values, levels), expected)


def test_quantile_crossing_repair_is_nonnegative_and_bounded():
    result = mixture_median([0.75], [[4, -1, 3, 2]], [0.05, 0.1, 0.25, 0.5])
    assert 0 <= result[0] <= 4


@pytest.mark.parametrize("p", [[np.nan], [-0.1], [1.1]])
def test_invalid_probability_fails(p):
    with pytest.raises(ValueError):
        mixture_median(p, [[1, 2]], [0.1, 0.5])
