"""Analytical probability transforms only; no calibrator fitting."""

import numpy as np
import pytest

from src.models.probability_calibration import calibrated_probability, probability_logits


def test_identity_transform_and_finite_endpoints():
    probability = np.array([0, 0.1, 0.5, 0.9, 1])
    np.testing.assert_allclose(calibrated_probability(probability, 1, 0), np.clip(probability, 1e-6, 1-1e-6))
    assert np.isfinite(probability_logits(probability)).all()


def test_stable_sigmoid_and_known_intercept():
    np.testing.assert_allclose(calibrated_probability([0.5], 1, np.log(3)), [0.75])
    assert calibrated_probability([0.5], 1, 1000)[0] == 1
    assert calibrated_probability([0.5], 1, -1000)[0] == 0


@pytest.mark.parametrize("p", [[np.nan], [-0.01], [1.01]])
def test_bad_probabilities_fail(p):
    with pytest.raises(ValueError):
        probability_logits(p)
