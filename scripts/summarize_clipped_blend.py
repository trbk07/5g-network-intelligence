"""Replay fixed cloud choices and the stricter current-research-control gate."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.select_clipped_blend import strict_acceptance
from scripts.summarize_development_blend import verify_blend, verify_manifest
from src.data.io import sha256
from src.utils.config import load_config


def main():
    cfg = load_config("configs/clipped_blend.yaml")
    sources = {"clipped": Path("artifacts/mae_aligned/direct_clipped"),
               "hurdle": Path("artifacts/hurdle_research"), "current": Path("artifacts/mlp_blend")}
    for name, path in sources.items():
        if sha256(path/"manifest.json") != cfg[name+"_manifest_sha256"]:
            raise ValueError("Pinned source manifest differs")
        verify_manifest(path)
    verify_blend("configs/clipped_blend.yaml", "artifacts/clipped_blend", sources["clipped"],
                 "clipped_blend", "clipped")
    root = Path("artifacts/clipped_blend")
    confirmation = pd.read_csv("reports/clipped_blend_confirmation.csv")
    rows = []
    for row in confirmation.to_dict("records"):
        h = row["horizon"]
        with np.load(root/f"confirmation_h{h}.npz", allow_pickle=False) as new, np.load(
                sources["current"]/f"confirmation_h{h}.npz", allow_pickle=False) as old:
            for key in ("actual", "origins", "beam_ids"):
                np.testing.assert_array_equal(new[key], old[key])
            mask = np.isfinite(old["actual"])
            reference = float(np.abs(old["predicted"][mask]-old["actual"][mask]).mean())
        rows.append({**row, "current_mae": reference,
                     "relative_mae_gain_vs_current_pct": 100*(1-row["mae"]/reference)})
    frame = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(frame, pd.read_csv("reports/clipped_blend_current_control.csv"),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    decision = json.loads((root/"decision.json").read_text())
    expected = bool(decision["passes_declared_development_confirmation"] and strict_acceptance(
        rows, cfg["minimum_relative_gain_vs_current_at_one_horizon"]))
    if (decision["passes_strict_current_control_gate"] != expected
            or decision["current_source_manifest_sha256"] != cfg["current_manifest_sha256"]):
        raise ValueError("Strict control decision differs from verified saved forecasts")
    verification_path = Path("reports/clipped_blend_verification.json")
    verification = json.loads(verification_path.read_text())
    verification.update(current_control_metrics_reconstructed=True,
                        passes_strict_current_control_gate=expected, production_promoted=False)
    verification_path.write_text(json.dumps(verification, indent=2))
    print(frame.to_string(index=False))
    print("Passes strict current-control gate:", expected)


if __name__ == "__main__":
    main()
