"""Deterministic risk statuses with explicit exclusion and collapsed-band handling."""

import numpy as np

from src.risk.thresholds import Thresholds


def risk_status(pred: np.ndarray, threshold: Thresholds) -> np.ndarray:
    """Normal <P90, Warning >=P90 and <P95, Critical >=P95; no Warning when P90=P95."""
    pred = np.asarray(pred)
    if pred.shape[-1] != len(threshold.beam_ids):
        raise ValueError("Risk beam axis differs")
    valid = np.isfinite(pred) & threshold.valid
    status = np.full(pred.shape, "Excluded", dtype="U8")
    status[valid] = "Normal"
    status[valid & (pred >= threshold.warning) & (pred < threshold.critical)] = "Warning"
    status[valid & (pred >= threshold.critical)] = "Critical"
    return status
