"""Summarize verified transfer evidence; descriptive statistics only, never fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records


def main():
    reports = Path("reports")
    single = risk_records(reports / "forecasts/transfer_pilot_val.npz")
    multi = risk_records(reports / "forecasts/multisource_pilot_val.npz")
    rows = []
    for h in (1, 6, 24):
        control = multi[h]["target_only"]
        # Two separately executed jobs must reproduce identical controls and shared cases.
        for left, right in (("transfer_0", "target_only"), ("transfer_0.5", "pm01_nr1800")):
            for field in ("actual", "origins", "predicted"):
                np.testing.assert_array_equal(single[h][left][field], multi[h][right][field])
        for name, record in multi[h].items():
            np.testing.assert_array_equal(record["actual"], control["actual"])
            np.testing.assert_array_equal(record["origins"], control["origins"])
            assert record["beam_ids"] == control["beam_ids"]
            actual = control["actual"]
            mask = np.isfinite(actual)
            if not np.isfinite(record["predicted"][mask]).all():
                raise ValueError("Missing predictions must not improve apparent coverage")
            base = np.abs(control["predicted"] - actual)
            errors = np.abs(record["predicted"] - actual)
            delta = errors - base
            # Blocks describe temporal consistency, not independent trials or significance.
            days = control["origins"] // 24
            daily = [float(delta[days == d][mask[days == d]].mean()) for d in np.unique(days)]
            rows.append({
                "model": name, "horizon": h,
                "mae": float(errors[mask].mean()),
                "mae_change_vs_control": float(delta[mask].mean()),
                "relative_mae_gain_pct": float(-100 * delta[mask].mean() / base[mask].mean()),
                "wape": float(errors[mask].sum() / np.abs(actual[mask]).sum()),
                "origin_day_blocks": len(daily),
                "blocks_with_lower_mae": sum(d < 0 for d in daily),
                "n_valid": int(mask.sum()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(reports / "transfer_ablation_comparison.csv", index=False)
    (reports / "transfer_ablation_decision.md").write_text(
        "# External transfer ablation decision\n\n"
        "Two independent hosted executions reproduce the target-only control and the shared NR1800 case exactly at all horizons. "
        "Checkpoint checksums and 81 history-only CPU/cloud comparisons passed; maximum difference is zero.\n\n"
        "NR1800 causes negative transfer at every tested weight (0.1, 0.5, 1). The balanced pool also regresses at all horizons. "
        "NR3500 has only small descriptive gains; these do not establish generalization or justify promotion. "
        "The source bands come from the same published collection and are not independent networks.\n\n"
        "```csv\n" + table.to_csv(index=False) + "```\n\n"
        "Day blocks group forecast origins, with a partial boundary day. They overlap in target times at longer horizons "
        "and are descriptive, not independent statistical replicates. Reused validation remains exploratory.\n\n"
        "Decision: preserve frozen production; do not increase external weight or pool by volume alone. "
        "Any further transfer candidate needs development-only temporal folds and source-specific controls before a new frozen evaluation. "
        "WAPE remains about 41–45%; the <10% goal is not achieved. No original final holdout was opened.\n",
        encoding="utf-8",
    )
    ledger = json.loads((reports / "transfer_run_ledger.json").read_text())
    ledger["attempts"][-1]["status"] = "complete_downloaded_checksum_and_cpu_parity_verified"
    ledger["multisource"] = {"version": 1, "status": "complete_downloaded_checksum_and_cpu_parity_verified"}
    (reports / "transfer_run_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
