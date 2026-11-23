"""Fixed numerical accounting examples; no model fitting."""

import numpy as np
import pytest

from scripts.audit_development_errors import decompose


def test_zero_and_positive_errors_reconcile_with_missing_targets():
    row = decompose(np.array([0., 0., 2., 4., np.nan]), np.array([1., 0., 1., 6., np.nan]))
    assert row["n_valid"] == 4
    assert row["zero_absolute_error"] == 1
    assert row["positive_absolute_error"] == 3
    assert row["mae"] == 1
    assert row["positive_signed_bias"] == .5
    assert row["positive_underprediction_fraction"] == .5


@pytest.mark.parametrize("actual,prediction", [([0, 1], [0, np.nan]), ([-1], [0]), ([1, 2], [1])])
def test_invalid_forecasts_cannot_silently_drop_observed_targets(actual, prediction):
    with pytest.raises(ValueError):
        decompose(actual, prediction)
