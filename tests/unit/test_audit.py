import numpy as np
import pytest

from src.data.audit import (
    audit_beams, audit_missing, audit_range, audit_zeros, check_temporal_boundary,
)
from src.data.panel import Panel
from src.data.splits import build_split_config


def panel(n):
    return Panel(np.zeros((n, 1, 4)), np.arange(n), ("0_0_0",), {})


def test_observed_denominator_and_range():
    data = panel(4)
    data.values[:, 0, 0] = [0, 2, np.nan, -1]
    assert audit_missing(data)["per_kpi"]["DLPRB"] == .25
    assert audit_zeros(data)["DLPRB"]["zero_ratio"] == pytest.approx(1/3)
    assert audit_zeros(data)["DLPRB"]["active_ratio"] == pytest.approx(1/3)
    assert audit_range(data)["DLPRB"]["negative_count"] == 1
    assert not audit_beams(data)["complete_expected_mapping"]


def test_boundary_requires_evidence_and_structure():
    assert not check_temporal_boundary(panel(840), panel(168))["continuity_verified"]
    assert not check_temporal_boundary(panel(840), panel(168), verified=True)[
        "continuity_verified"]
    assert check_temporal_boundary(panel(840), panel(168), verified=True, evidence="card")[
        "continuity_verified"]
    assert not check_temporal_boundary(panel(839), panel(168), verified=True, evidence="card")[
        "continuity_verified"]


@pytest.mark.parametrize("primary,n,fit", [(False, 840, 503), (True, 1008, 671)])
def test_splits(primary, n, fit):
    split = build_split_config(n, primary)
    assert split["fit"] == [0, fit]
    assert split["validation"] == [fit+1, fit+168]
    assert split["holdout"] == [fit+169, n-1]
    assert split["final_refit"][1] < split["holdout"][0]
    with pytest.raises(ValueError):
        build_split_config(n-1, primary)
