"""Read-only error decomposition; no blend weights, fitting or model selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.select_development_blend import verify_manifest


def main():
    roots = {"clipped": Path("artifacts/mae_aligned/direct_clipped"),
             "mlp": Path("artifacts/temporal_mlp"), "hurdle": Path("artifacts/hurdle_research")}
    for root in roots.values():
        verify_manifest(root)
    rows = []
    for fold in ("inner1", "inner2"):
        for h in (1, 6, 24):
            paths = {"clipped": roots["clipped"]/fold/"B"/f"score_h{h}.npz",
                     "mlp": roots["mlp"]/fold/"B"/f"score_h{h}.npz",
                     "hurdle": roots["hurdle"]/fold/f"h{h}"/"score.npz"}
            arrays = {}
            reference = None
            for name, path in paths.items():
                with np.load(path, allow_pickle=False) as score:
                    axes = {k: score[k].copy() for k in ("actual", "origins", "beam_ids")}
                    if reference is None:
                        reference = axes
                    else:
                        for key in axes:
                            np.testing.assert_array_equal(axes[key], reference[key])
                    arrays[name] = score["hurdle_median" if name == "hurdle" else "predicted"].copy()
            actual = reference["actual"]
            mask = np.isfinite(actual)
            residuals = {}
            for name, prediction in arrays.items():
                if not np.isfinite(prediction[mask]).all():
                    raise ValueError("Diagnostic must preserve the full observed population")
                residuals[name] = prediction[mask]-actual[mask]
                for segment, selected in (("zero", mask & (actual == 0)),
                                           ("positive", mask & (actual > 0))):
                    rows.append({"fold": fold, "horizon": h, "model": name, "segment": segment,
                        "n_valid": int(selected.sum()),
                        "mae": float(np.abs(prediction[selected]-actual[selected]).mean()),
                        "predicted_zero_rate": float((prediction[selected] == 0).mean()),
                        "clipped_residual_correlation": float(np.corrcoef(residuals["clipped"], residuals[name])[0, 1]),
                        "clipped_opposite_error_sign_fraction": float((residuals["clipped"]*residuals[name] < 0).mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv("reports/clipped_forecast_error_diagnostics.csv", index=False)
    Path("reports/clipped_forecast_diagnostic_scope.json").write_text(json.dumps({
        "rows": len(frame), "axes_and_manifests_verified": True,
        "segmentation": "observed zeros and positive targets; no fitted threshold",
        "correlation_scope": "full observed population repeated alongside each segment",
        "blend_weights_computed": False, "training_performed": False,
        "holdout_used": False, "validation_504_671_used": False,
        "claim": "error structure only; no proof that a blend improves MAE"}, indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
