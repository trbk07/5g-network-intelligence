"""Hosted-only coefficient selection on inner1 and fixed confirmation on inner2."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.models.forecast_blend import blend_forecasts
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def verify_manifest(root):
    files = json.loads((root/"manifest.json").read_text())["files"]
    for name, digest in files.items():
        if sha256(root/name) != digest:
            raise ValueError("Attached source checkpoint checksum mismatch")


def forecasts(linear_root, hurdle_root, fold, variant, h):
    linear_folder, hurdle_folder = linear_root/fold/variant, hurdle_root/fold/f"h{h}"
    lm = json.loads((linear_folder/"meta.json").read_text())
    hm = json.loads((hurdle_folder/"meta.json").read_text())
    if (lm["fold"] != hm["fold"] or hm["fold"]["score_end"] > 503
            or lm["holdout_used"] or hm["holdout_used"]
            or lm["validation_504_671_used"] or hm["validation_504_671_used"]):
        raise ValueError("Source temporal protocol differs")
    with np.load(linear_folder/f"score_h{h}.npz", allow_pickle=False) as linear, np.load(hurdle_folder/"score.npz", allow_pickle=False) as hurdle:
        for key in ("actual", "origins", "beam_ids"):
            np.testing.assert_array_equal(linear[key], hurdle[key])
        return {"actual": linear["actual"].copy(), "origins": linear["origins"].copy(),
                "beam_ids": linear["beam_ids"].copy(), "hurdle": hurdle["hurdle_median"].copy(),
                "linear": linear["predicted"].copy()}


def metrics(record, weight):
    prediction = blend_forecasts(record["hurdle"], record["linear"], weight)
    actual = record["actual"]
    mask = np.isfinite(actual)
    error = np.abs(prediction[mask]-actual[mask])
    return prediction, {"mae": float(error.mean()), "wape": float(error.sum()/np.abs(actual[mask]).sum()), "n_valid": int(mask.sum())}


def run_selection(cfg, linear_root, hurdle_root, root, report_prefix, component="linear"):
    """Shared hosted procedure; component names remain explicit in saved evidence."""
    require_cloud_training()
    verify_manifest(linear_root)
    verify_manifest(hurdle_root)
    root = Path(root)
    weight_key = f"{component}_weight"
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite an existing selection")
    root.mkdir(parents=True,exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    grid_rows, choices, confirmation = [], {}, []
    # Choose every horizon before reading any confirmation forecast/target arrays.
    for h in cfg["horizons"]:
        candidates = []
        for variant in cfg["variants"]:
            record = forecasts(linear_root,hurdle_root,cfg["selection_fold"],variant,h)
            for weight in cfg[weight_key+"_grid"]:
                _, result = metrics(record,weight)
                row = {"fold":cfg["selection_fold"],"horizon":h,"variant":variant,weight_key:weight,**result}
                grid_rows.append(row)
                candidates.append(row)
        best = min(candidates,key=lambda row:(row["mae"],row[weight_key],cfg["variants"].index(row["variant"])))
        choices[str(h)] = {"variant":best["variant"],weight_key:best[weight_key]}
    selection = {"protocol":cfg,"choices":choices,"selection_data_only":cfg["selection_fold"],
                 "source_commit":os.environ.get("SOURCE_GIT_COMMIT"),
                 f"{component}_source_manifest_sha256":sha256(linear_root/"manifest.json"),
                 "hurdle_source_manifest_sha256":sha256(hurdle_root/"manifest.json")}
    (root/"selection.json").write_text(json.dumps(selection,indent=2))
    pd.DataFrame(grid_rows).to_csv(f"reports/{report_prefix}_selection_grid.csv",index=False)
    for h in cfg["horizons"]:
        choice = choices[str(h)]
        record = forecasts(linear_root,hurdle_root,cfg["confirmation_fold"],choice["variant"],h)
        prediction, result = metrics(record,choice[weight_key])
        _, baseline = metrics(record,0)
        confirmation.append({"fold":cfg["confirmation_fold"],"horizon":h,**choice,**result,
                             "hurdle_mae":baseline["mae"],"relative_mae_gain_pct":100*(1-result["mae"]/baseline["mae"])})
        np.savez_compressed(root/f"confirmation_h{h}.npz",actual=record["actual"],predicted=prediction,
                            hurdle=record["hurdle"],origins=record["origins"],beam_ids=record["beam_ids"],**{component:record["linear"]})
    acceptable = all(r["mae"] <= r["hurdle_mae"] for r in confirmation) and any(r[weight_key] > 0 and r["mae"] < r["hurdle_mae"] for r in confirmation)
    (root/"decision.json").write_text(json.dumps({"passes_declared_development_confirmation":acceptable,
        "no_validation_audit_authorized_by_this_job":True,"holdout_used":False,
        "claim_scope":"reused development only, no generalization or promotion claim"},indent=2))
    pd.DataFrame(confirmation).to_csv(f"reports/{report_prefix}_confirmation.csv",index=False)
    files = {p.relative_to(root).as_posix():sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files":files},indent=2))
    print(json.dumps(choices,indent=2),flush=True)
    print(pd.DataFrame(confirmation).to_string(index=False),flush=True)
    print("Passes fixed development confirmation:",acceptable,flush=True)



def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-root", type=Path, required=True)
    parser.add_argument("--hurdle-root", type=Path, required=True)
    args = parser.parse_args()
    run_selection(load_config("configs/development_blend.yaml"),args.linear_root,args.hurdle_root,
                  "artifacts/development_blend","development_blend")


if __name__ == "__main__":
    main()
