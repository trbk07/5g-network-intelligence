"""Fixed-coefficient forecast arithmetic only; no fitting or selection."""

import numpy as np


def blend_forecasts(hurdle, linear, linear_weight):
    hurdle, linear = np.asarray(hurdle), np.asarray(linear)
    if hurdle.shape != linear.shape or not np.isfinite(linear_weight) or not 0 <= linear_weight <= 1:
        raise ValueError("Aligned forecasts and scalar weight in [0,1] required")
    if not np.isfinite(hurdle).all() or not np.isfinite(linear).all():
        raise ValueError("Finite forecasts required")
    # Preserve original stored precision: float64 mixing is declared for reproducibility.
    return (1-linear_weight)*hurdle.astype(float) + linear_weight*linear.astype(float)
