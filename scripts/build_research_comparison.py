"""Descriptive comparison of saved validation replays; no fitting or model promotion."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records


SOURCES={
    "original":"lightgbm_val.npz",
    "ensemble":"improvement_audit.npz",
    "single_source":"transfer_pilot_val.npz",
    "multisource":"multisource_pilot_val.npz",
    "hurdle":"hurdle_candidate_val.npz",
}


def main():
    reports=Path("reports")
    reference=risk_records(reports/"forecasts/lightgbm_val.npz")
    rows=[]
    for family,filename in SOURCES.items():
        for h,group in risk_records(reports/"forecasts"/filename).items():
            expected=reference[h]["lightgbm_c1"]
            for name,record in group.items():
                np.testing.assert_array_equal(record["origins"],expected["origins"])
                np.testing.assert_array_equal(record["actual"],expected["actual"])
                if record["beam_ids"]!=expected["beam_ids"]:
                    raise ValueError("Research comparison requires identical original beam axes")
                observed=np.isfinite(record["actual"])
                if not np.isfinite(record["predicted"][observed]).all():
                    raise ValueError("Missing predictions cannot improve a research comparison")
                error=np.abs(record["predicted"][observed]-record["actual"][observed])
                rows.append({"family":family,"model":name,"horizon":h,"mae":float(error.mean()),
                             "wape":float(error.sum()/np.abs(record["actual"][observed]).sum()),
                             "n_valid":int(observed.sum()),"n_beams":len(record["beam_ids"]),
                             "n_origins":len(record["origins"]),"saved_replay":filename})
    frame=pd.DataFrame(rows)
    frame.to_csv(reports/"research_validation_comparison.csv",index=False)
    lines=["# Research validation comparison","",
        "Scope: selected saved replay families used by the current research extension. Original baseline/GRU release benchmarks remain in experiment_report.md; this table is not a replacement for that release evaluation.","",
        "All entries below are reconstructed from saved cloud validation forecasts. Origins, targets and beam axes match original C1 exactly; every observed target has a finite prediction. No training, coefficient search, final holdout evaluation or production promotion occurs in this report.","",
        "Validation 504–671 has been reused repeatedly. These are exploratory point estimates, not fresh-test evidence or a no-overfitting guarantee. MAE uses original dataset usage units; WAPE is displayed as a percentage of the same observed absolute-target sum. No target-unit conversion is assumed.","",
        "| Family / model | 1h MAE | 6h MAE | 24h MAE | 1h WAPE | 6h WAPE | 24h WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (family,name),group in frame.groupby(["family","model"],sort=False):
        if set(group.horizon)!={1,6,24} or len(group)!=3:
            raise ValueError("Every listed case must contain all three declared horizons")
        group=group.set_index("horizon")
        values=[f"{group.loc[h,'mae']:.6f}" for h in (1,6,24)]+[f"{100*group.loc[h,'wape']:.2f}%" for h in (1,6,24)]
        lines.append("| "+family+" / "+name+" | "+" | ".join(values)+" |")
    lines.extend(["","Interpretation:","",
        "- Hurdle improves all three point MAEs relative to the earlier improved ensemble. It is not the smallest point MAE at every horizon across all transfer experiments.",
        "- The NR3500 source case has lower 6h and 24h point MAE than hurdle, while hurdle has lower 1h MAE. The 24h difference is tiny and reused validation does not justify a new horizon-wise selector.",
        "- The target-only relative-shape control also has lower 6h point MAE than hurdle. Compare source cases with their matched control; extra dataset volume alone is not evidence of benefit.",
        "- Identical controls/shared source cases in separate jobs are replications, not independent datasets or new evidence of generalization.",
        "- The fixed neural blend passed its separately declared development confirmation. Its queued reused-validation audit is not included until the complete replay passes independent verification.",
        "- WAPE below 10% remains unmet. Production remains the frozen release; these research cases are not deployment replacements.","",
        "Evidence: [transfer decision](transfer_ablation_decision.md), [hurdle audit](hurdle_candidate_analysis.md), [calibration rejection](hurdle_calibration_analysis.md), [linear rejection](temporal_linear_analysis.md), [fixed blend confirmation](development_blend_analysis.md).",""])
    (reports/"research_validation_comparison.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Wrote {len(frame)} rows from {len(SOURCES)} saved replay families; all populations aligned. No model promoted.")


if __name__=="__main__":
    main()
