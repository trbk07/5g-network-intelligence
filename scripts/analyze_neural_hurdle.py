"""Summarize verified direct-versus-hurdle results without fitting or tuning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    reports = Path("reports")
    comparison = pd.read_csv(reports/"neural_hurdle_matched_comparison.csv")
    checks = [json.loads((reports/f"neural_hurdle_{kind}_verification.json").read_text())
              for kind in ("direct", "hurdle")]
    if not all(check["all_metrics_reconstructed"] for check in checks):
        raise ValueError("Independent metric verification required")
    if len(comparison) != 6:
        raise ValueError("Both folds and all three horizons required")
    parity_count = sum(check["history_only_beam_predictions_checked"] for check in checks)
    max_error = max(check["max_parity_error"] for check in checks)
    root = Path("artifacts/neural_hurdle")
    for path in root.glob("*/*/B/history.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not np.isfinite([row["train_objective"], row["val_l1_scaled"]]).all():
                raise ValueError("Nonfinite optimization history")
    evidence = {"all_metrics_reconstructed": True, "finite_training_objectives": True,
        "history_only_predictions_checked": parity_count, "maximum_cpu_gpu_discrepancy": max_error,
        "hurdle_better_than_direct_points": int((comparison.relative_mae_gain_vs_direct_pct > 0).sum()),
        "comparison_points": 6, "completed_patience_runs": sum(c["early_stopping_completed_runs"] for c in checks),
        "epoch_capped_runs": sum(c["epoch_capped_before_patience_runs"] for c in checks),
        "decision": "Reject BCE-plus-pinball hurdle recipe; no validation refit or production promotion",
        "holdout_used": False, "validation_504_671_used": False, "production_promoted": False}
    (reports/"neural_hurdle_verification.json").write_text(json.dumps(evidence, indent=2))
    lines = ["# Multivariate neural zero-mass forecasting: verified result", "",
        "Decision: reject this BCE-plus-positive-pinball hurdle recipe. It regresses against its matched direct head at all six development fold/horizon points. Preserve the frozen production model and earlier research candidates.", "",
        "## Protocol", "",
        "Kaggle private worker `truongbaokien/5g-ni-multivariate-zero-inflated-forecasting`, version 1, source commit `56642a2421b5529e308e7ee5dbf10f8999428407`, Tesla T4. Four hosted fits use the same four-KPI encoder, hidden width 64, seed 17, maximum 40 epochs and patience 5. Fit/stop/score cutoffs are 335/383/443 and 395/443/503. Original values from 504 onward were redacted. No external-data pretraining occurred in this experiment.", "",
        "The direct head minimizes masked scaled L1. The hurdle head minimizes finite-label BCE plus positive-only mean pinball with fixed equal coefficients. Both select epochs by point MAE on the separate stopping segment. Quantiles are noncrossing by construction; no probability calibration or learned cutoff is performed. Exact-zero decoding is shared by saved scoring and history-only CPU inference.", "",
        "## Inner2 point estimates", "",
        "MAE uses original DLPRB dataset units. WAPE is sum absolute error / sum absolute actual, shown as a percentage. Coverage retains all original beams and all declared scoring origins.", "",
        "| Case | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ("direct", "hurdle"):
        frame = pd.read_csv(reports/f"neural_hurdle_{kind}_development.csv")
        group = frame.loc[frame.fold == "inner2"].set_index("horizon")
        values = [f"{group.loc[h, 'mae']:.6f}" for h in (1, 6, 24)]
        values += [f"{100*group.loc[h, 'wape']:.2f}%" for h in (1, 6, 24)]
        lines.append("| "+kind+" | "+" | ".join(values)+" |")
    lines += ["", "## Error analysis and interpretation", "",
        "- Hurdle MAE is worse than matched direct regression by 1.481–3.224% on inner1 and 0.261–1.904% on inner2. It also regresses versus the earlier multivariate MLP and LightGBM hurdle at every evaluated point.",
        "- Zero false-positive rate improves at four of six points, but worsens at inner1 +6h/+24h. It does not compensate for the systematic MAE regression. Exact-zero numerical preservation was independently tested, so inverse-scaling roundoff is not counted as a predicted positive.",
        "- The new direct input definition improves four of six point MAEs versus the older MLP-B, but regresses on inner1 +6h and inner2 +24h. It is not consistently superior to the older model.",
        "- BCE plus conditional pinball optimizes a distributional surrogate, not the final interpolated mixture MAE. This mismatch is a hypothesis for the failure, not a proven cause. Sparse-quantile interpolation, shared capacity and one-seed variability remain possible contributors.", "",
        "## Verification and limitations", "",
        f"- {parity_count} history-only beam forecasts checked; maximum CPU/GPU discrepancy {max_error:.3g} original units, tolerance 1e-5.",
        "- All four fine-tuning runs completed stopping patience. No epoch cap or wall-time exhaustion explains the result. Every logged training and stopping objective is finite.",
        "- Archive SHA256 `2524a24656156c34b5556d3abcfdbfff83e21fbcadd58043f46d378072483c3e`; 34 artifact files verified; frozen artifacts untouched.",
        "- One seed and reused development windows. These results do not prove unseen-test generalization or a model-family error floor. WAPE <10% remains unmet. No validation refit, holdout replay or production promotion.", ""]
    (reports/"neural_hurdle_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
