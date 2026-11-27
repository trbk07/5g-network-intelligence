"""Present verified objective ablations, retaining mixed and negative outcomes."""

import json
from pathlib import Path

import pandas as pd


def main():
    reports = Path("reports")
    modes = ("direct_clipped", "hurdle_mae")
    controls = pd.read_csv(reports/"mae_aligned_control_comparison.csv")
    checks = [json.loads((reports/f"mae_aligned_{mode}_verification.json").read_text()) for mode in modes]
    if not all(c["all_metrics_reconstructed"] for c in checks) or len(controls) != 12:
        raise ValueError("Complete verified objective ablations required")
    frames = {mode: pd.read_csv(reports/f"mae_aligned_{mode}_development.csv") for mode in modes}
    keys = ["fold", "variant", "horizon"]
    clipped = frames["direct_clipped"].set_index(keys).sort_index()
    old = pd.read_csv(reports/"neural_hurdle_direct_development.csv").set_index(keys).sort_index()
    zero_gain = old.zero_fp_rate-clipped.zero_fp_rate
    verification = {
        "all_metrics_reconstructed": True, "history_only_predictions_checked": sum(c["history_only_beam_predictions_checked"] for c in checks),
        "maximum_cpu_gpu_discrepancy": max(c["max_parity_error"] for c in checks),
        "completed_patience_runs": sum(c["early_stopping_completed_runs"] for c in checks),
        "epoch_capped_runs": sum(c["epoch_capped_before_patience_runs"] for c in checks),
        "hurdle_mae_better_than_old_surrogate_points": int((controls.loc[controls["mode"] == "hurdle_mae", "relative_mae_gain_vs_control_pct"] > 0).sum()),
        "direct_clipped_better_than_old_direct_points": int((controls.loc[controls["mode"] == "direct_clipped", "relative_mae_gain_vs_control_pct"] > 0).sum()),
        "direct_clipped_zero_fp_improved_points": int((zero_gain > 0).sum()),
        "decision": "No standalone promotion; objective changes yield mixed gains versus strong controls",
        "holdout_used": False, "validation_504_671_used": False, "production_promoted": False}
    (reports/"mae_aligned_verification.json").write_text(json.dumps(verification, indent=2))
    lines = ["# MAE-aligned objectives: verified development result", "",
        "Decision: retain the objective-alignment finding, but do not promote either standalone model or refit on final validation. The mixture-MAE objective improves the old surrogate hurdle consistently, without surpassing the stronger historical controls. Nonnegative direct training trades fewer zero false positives for mixed MAE effects.", "",
        "## Protocol and execution", "",
        "Kaggle private `truongbaokien/5g-ni-mae-aligned-forecasting`, version 1, source `50b92367a9796a0db569ea3b6fe9a0e810c3f675`. Four fits completed on Tesla T4 / Torch 2.11.0+cu130, matching the historical control environment. The Colab fallback was prepared but did not fit; its assignment was later reclaimed. No control was retrained.", "",
        "Encoder/head shapes, seed 17, four KPI inputs, batch size 1024 and maximum 40 epochs/patience 5 are unchanged. Fit/stop/score cutoffs: 335/383/443 and 395/443/503. The auxiliary coefficient 0.1 was fixed before training; no coefficient search used scoring. Original values from hour 504 onward were redacted.", "",
        "## Inner2 point estimates", "",
        "MAE is in original DLPRB dataset units. WAPE is sum absolute error / sum absolute actual, displayed as a percentage. These are reused development scores, not final validation or holdout.", "",
        "| Mode | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in modes:
        group = frames[mode].loc[frames[mode].fold == "inner2"].set_index("horizon")
        values = [f"{group.loc[h, 'mae']:.6f}" for h in (1, 6, 24)]
        values += [f"{100*group.loc[h, 'wape']:.2f}%" for h in (1, 6, 24)]
        lines.append("| "+mode+" | "+" | ".join(values)+" |")
    lines += ["", "## Findings and limitations", "",
        "- Mixture-MAE plus auxiliary losses improves all six MAEs versus the earlier BCE-plus-pinball hurdle: 1.085–2.500% on inner1 and 0.389–0.919% on inner2. This supports investigating objective alignment; it does not establish general superiority.",
        "- The point-optimized hurdle still loses to the historical LightGBM hurdle at every point, and to MLP-B at five of six points. Its zero false-positive rate is approximately 99.5–100%; a distributional head does not guarantee useful zero-mass discrimination. No probability/quantile calibration claim is permitted.",
        "- Direct-clipped improves three of six MAEs against matched old direct regression. Its +6h gain on inner1 reverses on inner2. Zero false-positive rate improves at all six points by "
        f"{100*zero_gain.min():.2f}–{100*zero_gain.max():.2f} percentage points, but this alone cannot justify replacing the stronger point forecaster.",
        f"- {verification['history_only_predictions_checked']} history-only beam predictions checked; maximum CPU/GPU discrepancy {verification['maximum_cpu_gpu_discrepancy']:.3g} original units (tolerance 1e-5). All four fits completed patience; no epoch cap.",
        "- Result archive SHA256 `28dc528915bee92784f52692d2ec7152056618b9d5d0ed48d82d294e4885df49`; 34 artifact files verified; frozen artifacts untouched.",
        "- One seed and repeatedly reused development windows. WAPE <10% remains unmet. No holdout evaluation, production promotion, independent-test superiority or model-family error-floor claim.", ""]
    (reports/"mae_aligned_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
