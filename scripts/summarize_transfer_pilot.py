"""Verify saved cloud transfer artifacts and compare original-network MAE/WAPE, without fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.io import sha256
from src.data.panel import load_panel
from src.models.transfer_forecaster import TransferForecaster


def main(kind):
    root = Path("artifacts") / kind
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(root / name) != digest:
            raise ValueError("Downloaded transfer artifact checksum mismatch")
    reports = Path("reports")
    forecasts = risk_records(reports / "forecasts" / f"{kind}_val.npz")
    reference = risk_records(reports / "forecasts/lightgbm_val.npz")
    panel = load_panel(Path("data/processed/panel.npz"))
    parity = []
    for h, group in forecasts.items():
        for name, record in group.items():
            np.testing.assert_array_equal(record["origins"], reference[h]["lightgbm_c1"]["origins"])
            np.testing.assert_array_equal(record["actual"], reference[h]["lightgbm_c1"]["actual"])
            if record["beam_ids"] != reference[h]["lightgbm_c1"]["beam_ids"]:
                raise ValueError("Original-network beam population differs")
            model = TransferForecaster(root / name / f"h{h}")
            for origin in (503, 551, 599):
                i = int(np.flatnonzero(record["origins"] == origin)[0])
                b = record["beam_ids"].index("0_0_0")
                pred = model.predict_at(
                    panel.values[origin - 167 : origin + 1, b : b + 1, :], origin, h
                )[0]
                error = abs(pred - record["predicted"][i, b])
                if error > 1e-8:
                    raise AssertionError("Transfer cloud/CPU history-only parity failed")
                parity.append(
                    {
                        "model": name,
                        "horizon": h,
                        "origin": origin,
                        "absolute_error": float(error),
                        "atol": 1e-8,
                    }
                )
    pd.DataFrame(parity).to_csv(reports / f"{kind}_cpu_parity.csv", index=False)
    frame = pd.read_csv(reports / f"{kind}_validation.csv")
    table = frame[frame.load_tier == "All"][["model", "horizon", "mae", "wape", "n_valid"]]
    # WAPE is comparable with MAE on exactly this fixed actual population; no metric rescaling.
    lines = [
        f"# {kind} - original-network transfer audit",
        "",
        "Cloud-only fitting. Original fit/stopping through 503; fixed refit and reused validation audit 504-671. No final holdout replay.",
        "Source domains are sector data, not additional original-network beams; relative-shape transfer does not assert physical-unit equivalence.",
        "",
        "```csv",
        table.to_csv(index=False).strip(),
        "```",
        "",
        "The external ratio-zero / target-only case is the transfer ablation control; compare source cases with it, not just with a differently featured original model.",
        "Target-only relative inputs drop incompatible exogenous KPIs for both domains. This isolates transfer under that representation; it does not prove the representation is better.",
        "All horizon/beam coverage must match original replay. Additional dataset volume is not evidence of higher quality.",
        "WAPE <10% is achieved only if every declared horizon meets it on a separately frozen unseen evaluation. Reused-validation results cannot establish that final claim.",
        f"CPU history-only parity checks: {len(parity)}; maximum absolute difference {max(r['absolute_error'] for r in parity):.3g}.",
        "",
    ]
    (reports / f"{kind}_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence = {
        "research_only": True,
        "holdout_used": False,
        "parity_checks": len(parity),
        "parity_max_abs_error": max(r["absolute_error"] for r in parity),
        "validation_wape_below_10pct_by_model": {
            name: bool(set(group.horizon) == {1, 6, 24} and (group.wape < 0.1).all())
            for name, group in table.groupby("model")
        },
        "claim_scope": "reused validation only, no final target achievement claim",
        "artifact_files_verified": len(manifest["files"]),
    }
    (reports / f"{kind}_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8", newline="\n"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["transfer_pilot", "multisource_pilot"], required=True)
    main(parser.parse_args().kind)
