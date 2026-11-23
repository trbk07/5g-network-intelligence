"""Probability scoring and labels only; no fitting or calibration."""

import numpy as np
import pytest

from scripts.train_risk_probability import labels_for
from scripts.summarize_risk_probability import verify_threshold_statistics
from src.data.panel import Panel
from src.risk.probability_metrics import probability_metrics
from src.risk.thresholds import Thresholds


def test_fold_threshold_labels_preserve_missing_and_excluded_beams():
    actual=np.array([[0.,2.,5.],[1.,np.nan,0.]])
    labels=labels_for(actual,np.array([1.,2.,1.]),np.array([True,True,False]))
    np.testing.assert_array_equal(labels,[0,1,np.nan,1,np.nan,np.nan])


def test_perfect_and_constant_probability_scores_have_known_values():
    perfect=probability_metrics([0,1],[0,1])
    assert perfect["brier"]==0
    assert perfect["average_precision"]==1
    assert perfect["f1"]==1
    control=probability_metrics([0,1],[.5,.5])
    assert control["brier"]==.25
    assert control["average_precision"]==.5
    assert control["log_loss"]==pytest.approx(np.log(2))


def test_no_events_are_flagged_as_undefined_not_perfect():
    result=probability_metrics([0,0],[.1,.2])
    assert result["average_precision"] is None
    assert result["recall"] is None
    assert "no_actual_positives" in result["flags"]


@pytest.mark.parametrize("labels,p", [([0,1],[0,np.nan]),([0,2],[.1,.9]),([0],[1.1])])
def test_invalid_observed_scores_fail(labels,p):
    with pytest.raises(ValueError):
        probability_metrics(labels,p)


def test_saved_threshold_verification_ignores_future_and_rejects_changed_state():
    # Analytical constant-prefix fixture, not a fitted threshold or model.
    values=np.ones((250,1,4))
    values[200:,:,0]=10000
    panel=Panel(values,np.arange(250),("0_0_0",),{})
    thresholds=Thresholds(np.array([1.]),np.array([1.]),np.array([True]),
        np.array(["eligible"]),np.array([1.]),np.array([200]),panel.beam_ids,199)
    policy={"quantiles":[.9,.95],"epsilon":1e-8,"min_observations":168,"min_active_ratio":.05}
    verify_threshold_statistics(panel,thresholds,{"fit_end":199},policy)
    thresholds.warning[0]=10000
    with pytest.raises(AssertionError):
        verify_threshold_statistics(panel,thresholds,{"fit_end":199},policy)
