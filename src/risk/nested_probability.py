"""Coherent nested-event probabilities by the conditional probability chain rule."""

import numpy as np


def nested_critical_probability(warning_probability, critical_given_warning):
    """P(critical)=P(warning)*P(critical|warning), without clipping or fitting."""
    warning=np.asarray(warning_probability,dtype=float)
    conditional=np.asarray(critical_given_warning,dtype=float)
    if (warning.shape!=conditional.shape or not np.isfinite(warning).all()
            or not np.isfinite(conditional).all()
            or ((warning<0)|(warning>1)|(conditional<0)|(conditional>1)).any()):
        raise ValueError("Aligned finite probabilities in [0,1] required")
    return warning*conditional
