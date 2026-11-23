"""Fixed arithmetic on stopping metadata only; no model fitting."""

import pytest

from scripts.finalize_blend_candidate import fixed_epochs


def metadata(best,completed=9,exhausted=False):
    return {"training":{"best_epoch":best,"epochs_completed":completed,"budget_exhausted":exhausted}}


def test_zero_based_best_epochs_are_converted_before_rounding():
    assert fixed_epochs([metadata(3),metadata(2)])==4
    assert fixed_epochs([metadata(1),metadata(2)])==3
    assert fixed_epochs([metadata(8,14),metadata(10,16)])==10


@pytest.mark.parametrize("sources", [[],[metadata(1)], [metadata(-1),metadata(2)],
    [metadata(2,completed=2),metadata(1)], [metadata(1,exhausted=True),metadata(2)]])
def test_incomplete_or_budget_limited_sources_are_rejected(sources):
    with pytest.raises(ValueError):
        fixed_epochs(sources)
