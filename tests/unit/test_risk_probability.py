"""Probability scoring and labels only; no fitting or calibration."""

import numpy as np
import pytest

from scripts.train_risk_probability import labels_for
from src.risk.probability_metrics import probability_metrics


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
