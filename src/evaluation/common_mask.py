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


def coverage_report(actual: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Separate raw missingness from exclusions caused by the shared comparison set."""
    if actual.shape != prediction.shape or mask.shape != actual.shape:
        raise ValueError("Coverage arrays must have identical shapes")
    individual = np.isfinite(actual) & np.isfinite(prediction)
    if np.any(mask & ~individual):
        raise ValueError("Common mask includes an unavailable pair")
    return {
        "n_origins": int(mask.any(1).sum()),
        "n_pairs": int(mask.sum()),
        "n_beams": int(mask.any(0).sum()),
        "n_total_beams": actual.shape[1],
        "coverage_pct": float(mask.mean() * 100) if mask.size else 0.0,
        "samples_excluded_pct": float((1 - mask.mean()) * 100) if mask.size else 100.0,
        "candidate_coverage_pct": float(individual.mean() * 100) if mask.size else 0.0,
        "n_missing_target": int((~np.isfinite(actual)).sum()),
        "n_missing_prediction": int((~np.isfinite(prediction)).sum()),
        "n_excluded_due_to_comparison": int((individual & ~mask).sum()),
    }
