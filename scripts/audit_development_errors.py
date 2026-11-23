"""Decompose fixed development forecasts; no fitting or coefficient selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.origins import evaluation_origins
from src.utils.config import load_config


def decompose(actual, prediction):
    """Keep all finite targets and split absolute error into zero/positive parts."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if actual.shape != prediction.shape or actual.ndim != 1:
        raise ValueError("Aligned flat forecasts required")
    valid = np.isfinite(actual)
    if not valid.any() or (actual[valid] < 0).any() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Finite forecasts and nonnegative observed targets required")
    y, p = actual[valid], prediction[valid]
    error = np.abs(p-y)
    zero = y == 0
    total = float(error.sum())
    denominator = float(y.sum())
    return {
        "n_valid": len(y), "zero_count": int(zero.sum()), "mae": float(error.mean()),
        "zero_target_fraction": float(zero.mean()),
        "zero_absolute_error": float(error[zero].sum()),
        "positive_absolute_error": float(error[~zero].sum()),
        "wape": total/denominator if denominator else None,
        "wape_if_only_zero_errors_were_perfect": float(error[~zero].sum()/denominator) if denominator else None,
        "zero_share_of_absolute_error": float(error[zero].sum()/total) if total else 0.0,
        "positive_signed_bias": float((p[~zero]-y[~zero]).mean()) if (~zero).any() else None,
        "positive_underprediction_fraction": float((p[~zero]<y[~zero]).mean()) if (~zero).any() else None,
        "zero_conditional_mae": float(error[zero].mean()) if zero.any() else None,
        "positive_conditional_mae": float(error[~zero].mean()) if (~zero).any() else None,
    }


def main():
    cfg = load_config("configs/temporal_mlp.yaml")
    rows = []
    for fold in cfg["folds"]:
        if fold["score_end"] > 503:
            raise ValueError("Development audit cannot access validation or holdout")
        for h in cfg["horizons"]:
            with np.load(Path("artifacts/hurdle_research")/fold["name"]/f"h{h}/score.npz", allow_pickle=False) as base:
                np.testing.assert_array_equal(base["origins"], evaluation_origins(fold["stop_end"], fold["score_end"], h))
                y = base["actual"]
                rows.append({"fold": fold["name"], "horizon": h, "model": "hurdle", **decompose(y, base["hurdle_median"])})
                for family in ("temporal_linear", "temporal_mlp"):
                    for variant in cfg["variants"]:
                        with np.load(Path("artifacts")/family/fold["name"]/variant/f"score_h{h}.npz", allow_pickle=False) as neural:
                            for key in ("actual", "origins", "beam_ids"):
                                np.testing.assert_array_equal(neural[key], base[key])
                            rows.append({"fold": fold["name"], "horizon": h, "model": f"{family}_{variant}",
                                         **decompose(y, neural["predicted"])})
    frame = pd.DataFrame(rows)
    np.testing.assert_allclose(frame.zero_absolute_error+frame.positive_absolute_error, frame.mae*frame.n_valid)
    frame.to_csv("reports/development_error_decomposition.csv", index=False)
    Path("reports/development_error_decomposition_protocol.json").write_text(json.dumps({
        "scope": "fixed forecasts on reused development windows only",
        "rows": len(rows), "model_fit_or_coefficient_selection": False,
        "error_components_reconcile": True, "validation_or_holdout_used": False,
        "units": "original target units; fractions in [0,1]",
        "warning": "Target-conditioned groups are diagnostic only; unavailable as inference-time routing labels"
    }, indent=2))
    print(frame[frame.model == "hurdle"].to_string(index=False))


if __name__ == "__main__":
    main()
