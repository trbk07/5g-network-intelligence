"""Pure aggregate arithmetic; no fitting, selection or calibration."""

import pandas as pd
import pytest

from scripts.assess_telecom_quality import gate, paired


def fixture():
    return pd.DataFrame([{"fold": f, "horizon": h, "mae": 1., "wape": .1, "n_valid": 100}
                         for f in ("inner1", "inner2") for h in (1, 6, 24)])


def test_large_mean_gain_does_not_hide_one_regression():
    reference = fixture()
    candidate = reference.copy()
    candidate["mae"] = [.8, .8, 1.01, .8, .8, .8]
    result = gate(paired(candidate, reference))
    assert result["mean_relative_mae_gain_pct"] > 5
    assert not result["initial_research_gate_passed"]
    assert not result["material_point_gain_gate_passed"]


def test_wape_threshold_is_strict_and_never_promotes():
    reference = fixture()
    candidate = reference.copy()
    candidate["mae"] = .9
    result = gate(paired(candidate, reference))
    assert result["material_point_gain_gate_passed"]
    assert not result["all_development_WAPE_below_10pct"]
    candidate["wape"] = .099
    result = gate(paired(candidate, reference))
    assert result["all_development_WAPE_below_10pct"]
    assert not result["production_promotion_permitted"]


def test_partial_or_mismatched_coverage_rejected():
    reference = fixture()
    with pytest.raises(ValueError, match="Complete unique"):
        paired(reference.iloc[:5], reference)
    candidate = reference.copy()
    candidate.loc[0, "n_valid"] = 99
    with pytest.raises(ValueError, match="Identical observed"):
        paired(candidate, reference)
    reference["mae"] = 0
    with pytest.raises(ValueError, match="positive reference"):
        paired(fixture(), reference)
