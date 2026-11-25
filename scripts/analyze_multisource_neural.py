"""Present verified multisource development evidence without fitting or selection."""

import json
from pathlib import Path

import pandas as pd


def main():
    reports = Path("reports")
    verification = json.loads((reports/"multisource_neural_verification.json").read_text())
    if not verification["all_five_cases_verified"]:
        raise ValueError("Complete independent verification required")
    frame = pd.read_csv(reports/"multisource_neural_ablation.csv")
    cases = list(frame.case.unique())
    parity = pd.concat([pd.read_csv(reports/f"multisource_neural_{case}_cpu_parity.csv") for case in cases])
    stops = pd.concat([pd.read_csv(reports/f"multisource_neural_{case}_training_termination.csv") for case in cases])
    latest = frame.loc[frame.fold == "inner2"]
    lines = ["# Multisource neural pretraining: verified development result", "",
        "Decision: do not promote this pretraining recipe or refit it on validation. More source data did not establish a consistent gain over the compute-matched target self-pretraining control. Preserve the frozen production model and the previous research candidates.", "",
        "## Protocol and provenance", "",
        "Kaggle private worker `truongbaokien/5g-ni-multisource-representation-pretraining`, version 3, completed with source commit `38f569c147e0203e5a9b5d412c0f98bc68560b07`. Versions 1 and 2 stopped before model creation due to verified input-layout/storage-detection errors; no completed fit was repeated to repair presentation.", "",
        "Fifteen hosted fits: three fixed external pretraining runs, two fold-specific target self-pretraining runs and ten target fine-tuning runs. Every pretraining run completed five epochs and exactly 320 optimizer steps. Fine-tuning used the same 64-unit univariate residual MLP, seed 17 and frozen fit/stop/score windows 335/383/443 and 395/443/503. All original beams and complete per-horizon scoring coverage were retained. Original values from 504 onward were redacted in the cloud bundle.", "",
        "Radio domains NR1800/NR2100/NR3500 are one published collection. CESNET adds a distinct IP-traffic collection, not PRB measurements. Pool sampling assigns half the mass to CESNET and half to radio. Global per-source prefix preprocessing, source checksums, transferred initialization tensors and optimizer step counts were independently verified.", "",
        "## Inner2 point estimates", "",
        "MAE is in original DLPRB dataset units; WAPE is a percentage of the same observed absolute-target sum. These are reused development results, not final validation or holdout estimates.", "",
        "| Case | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in cases:
        group = latest.loc[latest.case == case].set_index("horizon")
        values = [f"{group.loc[h, 'mae']:.6f}" for h in (1, 6, 24)]
        values += [f"{100*group.loc[h, 'wape']:.2f}%" for h in (1, 6, 24)]
        lines.append("| "+case+" | "+" | ".join(values)+" |")
    lines += ["", "## Negative transfer and controls", "",
        "| External case | Inner2 1h gain vs self | Inner2 6h gain vs self | Inner2 24h gain vs self |",
        "| --- | ---: | ---: | ---: |"]
    for case in ("radio", "cesnet", "radio_and_cesnet"):
        group = latest.loc[latest.case == case].set_index("horizon")
        values = [f"{group.loc[h, 'relative_mae_gain_vs_target_self_pretrain_pct']:+.4f}%" for h in (1, 6, 24)]
        lines.append("| "+case+" | "+" | ".join(values)+" |")
    lines += ["", "Positive means lower MAE; negative means regression. Radio improves five of six fold/horizon point MAEs versus the no-pretraining control, but it improves only one of three inner2 cases versus self-pretraining. CESNET-only and pooled effects are mixed. Additional updates explain part of the apparent benefit; dataset count alone is not a useful acceptance criterion.", "",
        "No external recipe is consistently better than the historical hurdle reference: each beats it only at inner2 +6h. The target-only case exactly reproduces the earlier univariate MLP scores at all six fold/horizon points. These observations do not establish an error floor or rule out a different transfer architecture.", "",
        "## Verification and limitations", "",
        f"- History-only CPU/GPU parity: {int(parity.beams_checked.sum())} beam forecasts, maximum absolute discrepancy {parity.max_error.max():.3g} original units (tolerance 1e-5).",
        f"- Completed fine-tuning patience: {int(stops.early_stopping_completed.sum())}/{len(stops)}; capped runs: {int((stops.termination_reason == 'epoch_cap').sum())}.",
        "- Archive SHA256: `ee5d15f49406d8e42ce490214ef907bc514a6ebe599e6e9f012b48e7959b4632`; 115 artifact files verified. Frozen artifacts were not touched.",
        "- One seed, univariate inputs and repeatedly reused development windows. No fresh-test superiority, no no-overfitting guarantee, no WAPE <10% claim and no production promotion.",
        "- Preserve this negative result. A future architecture should address the original multivariate signals and zero/nonzero conditional distribution, with matched controls; do not simply add more pooled rows under this unchanged recipe.", ""]
    (reports/"multisource_neural_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis written: {len(frame)} verified rows, {int(parity.beams_checked.sum())} parity forecasts")


if __name__ == "__main__":
    main()
