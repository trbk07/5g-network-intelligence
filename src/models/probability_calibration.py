"""Pure inference for a saved affine-logit probability calibrator."""

import numpy as np


def probability_logits(probability, epsilon=1e-6):
    probability = np.asarray(probability, dtype=float)
    if (not 0 < epsilon < 0.5 or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Finite probabilities in [0,1] and valid clipping epsilon required")
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def calibrated_probability(probability, slope, intercept, epsilon=1e-6):
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("Finite scalar calibration coefficients required")
    logits = slope * probability_logits(probability, epsilon) + intercept
    return np.exp(-np.logaddexp(0, -logits))
