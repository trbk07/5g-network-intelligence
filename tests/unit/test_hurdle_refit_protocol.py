"""Round selection is arithmetic on development metadata, with no fitting."""

import pytest

from scripts.finalize_hurdle_candidate import component_rounds


def test_component_rounds_use_both_folds_with_explicit_rounding():
    assert component_rounds([{"rounds": {"active": 10, "direct": 30}},
                             {"rounds": {"active": 11, "direct": 40}}]) == {"active": 11, "direct": 35}


@pytest.mark.parametrize("metadata", [[], [{"rounds": {}}],
    [{"rounds": {"active": 0}}, {"rounds": {"active": 2}}],
    [{"rounds": {"active": 2}}, {"rounds": {"direct": 2}}]])
def test_invalid_development_round_sources_fail(metadata):
    with pytest.raises(ValueError):
        component_rounds(metadata)
