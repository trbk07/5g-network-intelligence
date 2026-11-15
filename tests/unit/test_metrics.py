import numpy as np
import pytest

from src.evaluation.metrics import (
    active_only_mae, assign_load_tiers, macro_mae, macro_wape, mae, rmse, wape, zero_fp_rate,
)


def test_hand_calculated_metrics():
    a = np.array([[1, 0], [3, np.nan]])
    p = np.array([[2, 2], [1, 9]])
    assert mae(a, p).value == pytest.approx(5 / 3)
    assert rmse(a, p).value == pytest.approx(np.sqrt(3))
    assert wape(a, p).value == 1.25
    assert macro_mae(a, p).value == 1.75
    assert macro_wape(a, p).value == .75
    assert macro_wape(a, p).flags == ("excluded_beams=1",)
    assert active_only_mae(a, p).value == 1.5
    assert zero_fp_rate(a, p).value == 1
    assert mae(a, p).n_valid == 3 and mae(a, p).n_excluded == 1
    assert active_only_mae(a, p).n_excluded == 2
    assert zero_fp_rate(a, p).n_excluded == 3


def test_undefined_and_missing_predictions():
    assert np.isnan(wape(np.zeros(2), np.ones(2)).value)
    assert wape(np.zeros(2), np.ones(2)).flags == ("zero_denominator",)
    r = mae(np.array([1, np.nan, 3]), np.array([np.nan, 2, 4]))
    assert (r.value, r.n_valid, r.n_excluded) == (1, 1, 2)
    assert mae(np.array([np.nan]), np.array([0])).flags == ("no_valid_pairs",)
    assert zero_fp_rate(np.ones(2), np.ones(2)).flags == ("no_zero_observations",)
    assert zero_fp_rate(np.zeros(3), np.array([-1, 0, 1])).value == pytest.approx(1 / 3)
    assert zero_fp_rate(np.zeros(3), np.array([0, .1, 1]), epsilon=.1).value == pytest.approx(1/3)
    with pytest.raises(ValueError):
        mae(np.ones(2), np.ones(3))


def test_macro_is_not_pooled_when_coverage_differs():
    a = np.array([[0, 0], [0, np.nan]])
    p = np.array([[0, 6], [0, np.nan]])
    assert mae(a, p).value == 2
    assert macro_mae(a, p).value == 3


def test_load_tiers_ties_and_missing():
    values = np.array([[0, 1, 2, 3, 4, 5, np.nan]])
    assert assign_load_tiers(values).tolist() == ["Low", "Low", "Mid", "Mid", "High", "High",
                                                 "Unknown"]
    assert assign_load_tiers(np.ones((4, 3))).tolist() == ["Low"] * 3
