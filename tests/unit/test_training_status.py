"""Pure metadata arithmetic; no fitting, backward or optimizer steps."""

import pytest

from src.models.training_status import stopping_status
from scripts.finalize_blend_candidate import fixed_epochs


CFG={"max_epochs":30,"early_stopping_patience":5}


def test_epoch_cap_is_distinct_from_wall_time_budget():
    training={"best_epoch":26,"epochs_completed":30,"budget_exhausted":False}
    status=stopping_status(training,CFG)
    assert status["termination_reason"]=="epoch_cap"
    assert status["stale_epochs"]==3
    assert not status["early_stopping_completed"]
    with pytest.raises(ValueError,match="patience"):
        fixed_epochs([{"training":training,"protocol":CFG}]*2)


def test_patience_completed_at_cap_still_is_a_valid_stopping_result():
    training={"best_epoch":24,"epochs_completed":30,"budget_exhausted":False}
    assert stopping_status(training,CFG)["early_stopping_completed"]
    assert fixed_epochs([{"training":training,"protocol":CFG}]*2)==25


def test_time_budget_and_incomplete_states_are_explicit():
    training={"best_epoch":8,"epochs_completed":10,"budget_exhausted":True}
    assert stopping_status(training,CFG)["termination_reason"]=="time_budget"
    training["budget_exhausted"]=False
    assert stopping_status(training,CFG)["termination_reason"]=="incomplete"
