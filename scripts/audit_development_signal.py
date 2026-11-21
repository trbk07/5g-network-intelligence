"""Development-only descriptive dependence checks, not fitted models or error bounds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.origins import train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.utils.config import load_config


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    a, b = x[valid], y[valid]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    values = panel.values[:504]
    target = values[:, :, 0]
    # Fit-only per-beam means support diagnostics; they are never model inputs.
    beam_mean = np.nanmean(target, axis=0)
    grand_mean = np.nanmean(target)
    valid = np.isfinite(target)
    total_ss = np.nansum((target - grand_mean) ** 2)
    between_ss = np.sum(valid.sum(axis=0) * (beam_mean - grand_mean) ** 2)
    summary = {
        "scope": "descriptive fit segment 0-503 only; no fitting, no holdout",
        "zero_fraction": float(np.mean(target[valid] == 0)),
        "between_beam_share_of_target_sum_squares": float(between_ss / total_ss),
        "warning": "Association is not forecasting skill, causation or an irreducible-error bound. Beam means are diagnostic only.",
    }
    rows = []
    for h in (1, 6, 24):
        origins = train_origins(503, h, stride=1)
        features = build_features(values, origins, h).reshape(len(origins), len(panel.beam_ids), -1)
        y = target[origins + h]
        yc = y - np.nanmean(y, axis=0)
        for i, name in enumerate(FEATURE_NAMES):
            x = features[:, :, i].astype(float)
            xc = x - np.nanmean(x, axis=0)
            rows.append({"horizon": h, "feature": name,
                         "pooled_pearson": correlation(x, y),
                         "within_beam_centered_pearson": correlation(xc, yc),
                         "n_pairs": int((np.isfinite(x) & np.isfinite(y)).sum())})
    pd.DataFrame(rows).to_csv("reports/development_feature_associations.csv", index=False)
    Path("reports/development_signal_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).query("feature == 'DLPRB_lag_0'").to_string(index=False))


if __name__ == "__main__":
    main()
