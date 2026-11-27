"""Hosted coefficient selection, then strict comparison with the existing blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.select_development_blend import run_selection, verify_manifest
from src.data.io import sha256
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def strict_acceptance(rows, minimum_gain):
    """Gate precomputed confirmation results; never choose coefficients here."""
    if {row["horizon"] for row in rows} != {1, 6, 24} or len(rows) != 3:
        raise ValueError("All three confirmation horizons required")
    if any(not np.isfinite([r["mae"], r["current_mae"]]).all() or r["current_mae"] <= 0 for r in rows):
        raise ValueError("Finite errors and positive reference errors required")
    return (all(r["mae"] <= r["current_mae"] for r in rows)
            and any(r["clipped_weight"] > 0 and 1-r["mae"]/r["current_mae"] >= minimum_gain for r in rows))


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clipped", "hurdle", "current"):
        parser.add_argument(f"--{name}-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config("configs/clipped_blend.yaml")
    for name in ("clipped", "hurdle", "current"):
        path = getattr(args, name+"_root")
        if sha256(path/"manifest.json") != cfg[name+"_manifest_sha256"]:
            raise ValueError("Pinned source manifest differs: "+name)
        verify_manifest(path)
    root = Path("artifacts/clipped_blend")
    # This call saves every inner1 choice before accessing inner2 numerical arrays.
    run_selection(cfg, args.clipped_root, args.hurdle_root, root, "clipped_blend", component="clipped")
    confirmation = pd.read_csv("reports/clipped_blend_confirmation.csv")
    rows = []
    for row in confirmation.to_dict("records"):
        h = row["horizon"]
        with np.load(root/f"confirmation_h{h}.npz", allow_pickle=False) as new, np.load(
                args.current_root/f"confirmation_h{h}.npz", allow_pickle=False) as current:
            for key in ("actual", "origins", "beam_ids"):
                np.testing.assert_array_equal(new[key], current[key])
            mask = np.isfinite(new["actual"])
            if not np.isfinite(current["predicted"][mask]).all():
                raise ValueError("Current control has incomplete coverage")
            reference = float(np.abs(current["predicted"][mask]-current["actual"][mask]).mean())
        rows.append({**row, "current_mae": reference,
                     "relative_mae_gain_vs_current_pct": 100*(1-row["mae"]/reference)})
    decision_path = root/"decision.json"
    decision = json.loads(decision_path.read_text())
    decision["passes_strict_current_control_gate"] = bool(
        decision["passes_declared_development_confirmation"]
        and strict_acceptance(rows, cfg["minimum_relative_gain_vs_current_at_one_horizon"]))
    decision["current_source_manifest_sha256"] = sha256(args.current_root/"manifest.json")
    decision_path.write_text(json.dumps(decision, indent=2))
    pd.DataFrame(rows).to_csv("reports/clipped_blend_current_control.csv", index=False)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name != "manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
