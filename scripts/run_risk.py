"""Historical risk validation; holdout mode requires the guarded holdout runner output."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel import Panel
from src.data.io import sha256
from src.inference.freeze import config_hash, verify_artifacts
from src.evaluation.common_mask import comparable_mask
from src.evaluation.validation import validation_inputs
from src.risk.evaluator import evaluate_exceedance
from src.risk.thresholds import fit_thresholds
from src.utils.config import load_config


def risk_records(path: Path) -> dict:
    """Read pickle-free replay records grouped by horizon/model."""
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for entry in json.loads(str(archive["metadata"]))["records"]:
            prefix = entry["prefix"]
            h = entry["horizon"]
            result.setdefault(h, {})[entry["model"]] = {
                "actual": archive[prefix + "_actual"],
                "predicted": archive[prefix + "_predicted"],
                "origins": archive[prefix + "_origins"],
                "beam_ids": tuple(archive[prefix + "_beam_ids"].tolist()),
            }
    return result


def main() -> None:
    """Compare all validation candidates and baselines on common masks; do not tune policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", choices=["val", "holdout"], default="val")
    args = parser.parse_args()
    data = load_config("configs/data.yaml")
    reports = Path(data["reports_dir"])
    if args.segment == "holdout":
        if not (reports / "holdout_completion.json").exists():
            raise RuntimeError("Guarded holdout has not completed")
        completed = json.loads((reports / "holdout_completion.json").read_text())
        frozen = load_config("configs/champion.yaml")
        artifacts = verify_artifacts()
        if completed["status"] != "complete" or completed["frozen_config_sha256"] != config_hash(
            frozen
        ):
            raise RuntimeError("Holdout completion provenance mismatch")
        if artifacts["freeze_commit"] != completed["freeze_commit"]:
            raise RuntimeError("Final artifact provenance mismatch")
        for name, digest in completed["files"].items():
            if sha256(Path(name)) != digest:
                raise RuntimeError(f"Holdout output mismatch: {name}")
        print("Verified saved holdout risk results; no new replay or training")
        return
    else:
        panel, split = validation_inputs(data)
        cfg = load_config("configs/risk.yaml")
        fit = Panel(
            panel.values[: split["fit"][1] + 1],
            panel.time[: split["fit"][1] + 1],
            panel.beam_ids,
            panel.metadata,
        )
        thresholds = fit_thresholds(fit, cfg)
        records = {}
        for filename in ("baselines_val.npz", "lightgbm_val.npz", "gru_val.npz"):
            for h, group in risk_records(reports / "forecasts" / filename).items():
                records.setdefault(h, {}).update(group)
    rows = []
    for h, group in records.items():
        actual = next(iter(group.values()))["actual"]
        origins = next(iter(group.values()))["origins"]
        for record in group.values():
            np.testing.assert_array_equal(actual, record["actual"])
            np.testing.assert_array_equal(origins, record["origins"])
            if record["beam_ids"] != thresholds.beam_ids:
                raise ValueError("Threshold beam axis differs")
        mask = comparable_mask(actual, {n: r["predicted"] for n, r in group.items()})
        for name, record in group.items():
            for row in evaluate_exceedance(
                actual, record["predicted"], thresholds, mask, cfg["min_positive_report"]
            ):
                rows.append({"segment": args.segment, "model": name, "horizon": h, **row})
    frame = pd.DataFrame(rows)
    suffix = "validation" if args.segment == "val" else "holdout"
    frame.to_csv(reports / f"risk_{suffix}.csv", index=False)
    if args.segment == "val":
        thresholds.save(reports / "thresholds_validation.npz")
        text = [
            "# Validation risk trade-off",
            "",
            "Policy fixed from fit units/sampling before validation risk metrics; no holdout-based tuning.",
            "P90/P95 are historical statistical scenarios, not operator thresholds or congestion labels.",
            f"Excluded beams: {int((~thresholds.valid).sum())}.",
            "No-positive / no-prediction metrics are undefined with flags, not 100% accuracy.",
            "",
            "```csv",
            frame.to_csv(index=False).strip(),
            "```",
        ]
        (reports / "risk_tradeoff_val.md").write_text("\n".join(text), encoding="utf-8")
    print("Risk evaluation:", args.segment, "; rows:", len(frame))


if __name__ == "__main__":
    main()
