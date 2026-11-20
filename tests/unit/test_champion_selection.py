import numpy as np
import pytest

from src.inference.registry import choose_candidate


def families(baseline_error, ml_error):
    actual = np.ones((5, 4))
    mask = np.ones_like(actual, dtype=bool)
    return {
        name: {h: (actual, actual + error, mask) for h in (1, 6, 24)}
        for name, error in (("persistence", baseline_error), ("lightgbm_c1", ml_error))
    }


def test_baseline_can_win_and_strict_ml_improvement():
    order = ["persistence", "lightgbm_c1"]
    assert (
        choose_candidate(families(1, 2), np.arange(4), order, repetitions=100)[0] == "persistence"
    )
    assert (
        choose_candidate(families(2, 1), np.arange(4), order, repetitions=100)[0] == "lightgbm_c1"
    )
    assert (
        choose_candidate(families(1, 1), np.arange(4), order, repetitions=100)[0] == "persistence"
    )


def test_family_requires_all_three_horizons():
    models = families(1, 2)
    del models["lightgbm_c1"][24]
    with pytest.raises(ValueError):
        choose_candidate(models, np.arange(4), list(models))


def test_cheaper_baseline_fallback_when_paired_ci_is_inconclusive():
    models = families(1.0, 1.4)
    for h in (1, 6, 24):
        actual, _, mask = models["persistence"][h]
        models["persistence"][h] = (actual, actual + np.array([0.0, 3.0, 0.0, 3.0]), mask)
    selected, rows = choose_candidate(models, np.arange(4), list(models), repetitions=1000)
    assert selected == "persistence"
    baseline = next(row for row in rows if row["family"] == "persistence")
    assert baseline["ci_lower_vs_best"] < 0 < baseline["ci_upper_vs_best"]
