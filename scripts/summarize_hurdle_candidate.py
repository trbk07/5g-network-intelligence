"""Verify fixed-refit cloud artifacts and compare validation records without fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.finalize_hurdle_candidate import component_rounds
from scripts.run_risk import risk_records
from scripts.summarize_hurdle_research import score_rows
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.models.hurdle_forecaster import HurdleForecaster
from src.utils.config import load_config


def main():
    reports = Path("reports")
    root = Path("artifacts/hurdle_candidate")
    forecaster = HurdleForecaster(root)
    cfg = load_config("configs/hurdle_research.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    replay = risk_records(reports / "forecasts/hurdle_candidate_val.npz")
    references = {
        "original_c1": (risk_records(reports / "forecasts/lightgbm_val.npz"), "lightgbm_c1"),
        "improved_ensemble": (risk_records(reports / "forecasts/improvement_audit.npz"), "improved_ensemble"),
    }
    rows, parity = [], []
    if set(replay) != {1, 6, 24}:
        raise ValueError("All declared horizons required")
    for h, group in replay.items():
        meta = forecaster.metadata[h]
        if meta["protocol"] != cfg:
            raise ValueError("Fixed-refit protocol mismatch")
        source = Path("artifacts/hurdle_research")
        if meta["source_manifest_sha256"] != sha256(source / "manifest.json"):
            raise ValueError("Development checkpoint source differs")
        development = [json.loads((source / fold["name"] / f"h{h}" / "meta.json").read_text()) for fold in cfg["folds"]]
        if meta["rounds"] != component_rounds(development):
            raise ValueError("Round counts differ from fixed development procedure")
        if set(group) != {"hurdle_direct_l1", "hurdle_median"}:
            raise ValueError("Both matched research cases required")
        origins = evaluation_origins(503, 671, h)
        actual = panel.values[origins + h, :, 0]
        for name, record in group.items():
            np.testing.assert_array_equal(record["origins"], origins)
            np.testing.assert_array_equal(record["actual"], actual)
            if record["beam_ids"] != panel.beam_ids:
                raise ValueError("Original full beam population required")
            rows.extend(score_rows(actual, {name: record["predicted"]}, "reused_validation", h))
            for origin in (503, 551, 599):
                index = int(np.flatnonzero(origins == origin)[0])
                for beam in ("0_0_0", "5_1_8", "15_2_0", "29_0_8"):
                    b = panel.beam_ids.index(beam)
                    prediction = float(forecaster.predict_at(panel.values[origin - 167:origin + 1, b:b + 1, :], origin, h, name)[0])
                    error = abs(prediction - record["predicted"][index, b])
                    if error > 1e-8:
                        raise AssertionError("History-only hurdle/cloud parity differs")
                    parity.append({"horizon": h, "model": name, "origin": origin, "beam": beam, "absolute_error": float(error)})
        for name, (reference, key) in references.items():
            record = reference[h][key]
            np.testing.assert_array_equal(record["actual"], actual)
            np.testing.assert_array_equal(record["origins"], origins)
            if record["beam_ids"] != panel.beam_ids:
                raise ValueError("Reference population differs")
            rows.extend(score_rows(actual, {name: record["predicted"]}, "reused_validation", h))
    frame = pd.DataFrame(rows)
    csv = pd.read_csv(reports / "hurdle_candidate_validation.csv")
    declared = csv.loc[csv.load_tier == "All"].set_index(["model", "horizon"])
    reconstructed = frame[frame.model.isin(("hurdle_direct_l1", "hurdle_median"))].set_index(["model", "horizon"])
    pd.testing.assert_frame_equal(reconstructed[["mae", "wape", "n_valid", "zero_fp_rate"]].sort_index(),
                                  declared[["mae", "wape", "n_valid", "zero_fp_rate"]].sort_index(),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    frame.to_csv(reports / "hurdle_candidate_comparison.csv", index=False)
    pd.DataFrame(parity).to_csv(reports / "hurdle_candidate_cpu_parity.csv", index=False)
    evidence = {"history_only_parity_checks": len(parity), "parity_max_error": max(r["absolute_error"] for r in parity),
                "all_metrics_reconstructed": True, "round_policy_verified": True,
                "holdout_used": False, "production_promoted": False,
                "claim_scope": "reused validation audit only; no unseen-test or overfitting guarantee"}
    (reports / "hurdle_candidate_verification.json").write_text(json.dumps(evidence, indent=2))
    (reports / "hurdle_candidate_analysis.md").write_text(
        "# Zero-inflated candidate audit\n\nFixed-round hosted refit; all original eligible beams. "
        "MAE is in original usage units and WAPE is a ratio.\n\n```csv\n" + frame.to_csv(index=False) + "```\n\n"
        "The matched direct L1 case tests the hurdle strategy; the earlier improved ensemble tests whether this candidate advances the best research result. "
        "Reused validation cannot establish performance on an independent future week. "
        "No production promotion, final holdout replay or probability calibration occurred.\n", encoding="utf-8")
    print(frame[["model", "horizon", "mae", "wape", "n_valid"]].to_string(index=False))


if __name__ == "__main__":
    main()
