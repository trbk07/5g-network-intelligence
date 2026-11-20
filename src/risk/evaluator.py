"""Evaluate >= historical exceedance with shared masks and undefined-metric flags."""

import numpy as np

from src.risk.thresholds import Thresholds


def evaluate_exceedance(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: Thresholds,
    mask: np.ndarray,
    min_positive_report: int = 10,
) -> list[dict]:
    """Return P90/P95 confusion counts, base rates and risk metrics without hiding undefined cases."""
    if y.shape != p.shape or y.shape != mask.shape or y.shape[1] != len(thresholds.beam_ids):
        raise ValueError("Risk arrays must share origin/beam axes")
    valid = mask & np.isfinite(y) & np.isfinite(p) & thresholds.valid[None, :]
    result = []
    for name, cutoff in (
        ("historical_warning_p90", thresholds.warning),
        ("historical_critical_p95", thresholds.critical),
    ):
        actual = (y >= cutoff)[valid]
        predicted = (p >= cutoff)[valid]
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        tn = int((~actual & ~predicted).sum())
        flags = []

        def ratio(numerator, denominator, flag):
            if denominator == 0:
                flags.append(flag)
                return float("nan")
            return numerator / denominator

        row = {
            "threshold": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "false_alarms": fp,
            "misses": fn,
            "n_valid": int(valid.sum()),
            "n_excluded": int(valid.size - valid.sum()),
            "n_excluded_beams": int((~thresholds.valid).sum()),
            "positive_events": tp + fn,
            "base_rate": ratio(tp + fn, tp + fp + fn + tn, "no_valid_pairs"),
            "precision": ratio(tp, tp + fp, "no_predicted_positives"),
            "recall": ratio(tp, tp + fn, "no_actual_positives"),
            "f1": ratio(2 * tp, 2 * tp + fp + fn, "undefined_f1"),
            "fpr": ratio(fp, fp + tn, "no_actual_negatives"),
        }
        if tp + fn < min_positive_report:
            flags.append("few_positive_events")
        row["flags"] = "|".join(flags)
        result.append(row)
    return result
