"""Probability arithmetic only; no fitting, calibration or threshold choice."""

import numpy as np
import pytest

from src.risk.nested_probability import nested_critical_probability


def test_chain_rule_has_known_probabilities_and_preserves_nesting():
    warning=np.array([0.,.2,.8,1.])
    conditional=np.array([1.,.5,.75,1.])
    result=nested_critical_probability(warning,conditional)
    np.testing.assert_allclose(result,[0.,.1,.6,1.])
    assert (result<=warning).all()
    np.testing.assert_array_equal(warning,[0.,.2,.8,1.])


@pytest.mark.parametrize("warning,conditional", [([.2],[np.nan]),([.2],[1.1]),([.2,.3],[.4])])
def test_invalid_probabilities_fail_without_clipping(warning,conditional):
    with pytest.raises(ValueError):
        nested_critical_probability(warning,conditional)
