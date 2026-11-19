"""Lock exactly the same finite observation pairs across a comparison set."""

import numpy as np


def comparable_mask(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Require finite actual and every candidate forecast; history NaNs do not exclude pairs."""
    actual = np.asarray(y_true)
    if actual.ndim != 2 or not predictions:
        raise ValueError("Expected (origin,beam) actuals and at least one candidate")
    mask = np.isfinite(actual)
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.shape != actual.shape:
            raise ValueError(f"Candidate shape differs: {name}")
        mask &= np.isfinite(prediction)
    return mask
