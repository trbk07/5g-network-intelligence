"""Verify cloud-selected coefficients and fixed confirmation; never reselect weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.models.forecast_blend import blend_forecasts
from src.utils.config import load_config


def verify_manifest(root):
    files = json.loads((root/"manifest.json").read_text())["files"]
    for name,digest in files.items():
        if sha256(root/name) != digest:
            raise ValueError("Source/artifact checksum mismatch")
    return len(files)


def verify_blend(config_path="configs/development_blend.yaml", artifact_path="artifacts/development_blend",
                 neural_path="artifacts/temporal_linear", report_prefix="development_blend", component="linear"):
    """Replay fixed cloud choices; no coefficient search is executed here."""
    cfg=load_config(config_path)
    root=Path(artifact_path)
    linear_root,hurdle_root=Path(neural_path),Path("artifacts/hurdle_research")
    weight_key=f"{component}_weight"
    file_count=verify_manifest(root)
    verify_manifest(linear_root)
    verify_manifest(hurdle_root)
    selection=json.loads((root/"selection.json").read_text())
    if (selection["protocol"]!=cfg or selection["selection_data_only"]!=cfg["selection_fold"]
            or selection[f"{component}_source_manifest_sha256"]!=sha256(linear_root/"manifest.json")
            or selection["hurdle_source_manifest_sha256"]!=sha256(hurdle_root/"manifest.json")
            or set(selection["choices"])!={str(h) for h in cfg["horizons"]}):
        raise ValueError("Stored selection protocol/source mismatch")
    grid=pd.read_csv(f"reports/{report_prefix}_selection_grid.csv")
    expected={(h,v,w) for h in cfg["horizons"] for v in cfg["variants"] for w in cfg[weight_key+"_grid"]}
    found=set(zip(grid.horizon,grid.variant,grid[weight_key]))
    if found!=expected or len(grid)!=len(expected) or set(grid.fold)!={cfg["selection_fold"]}:
        raise ValueError("Missing or duplicated selection-grid case")
    rows=[]
    checked=0
    max_error=0.
    for h in cfg["horizons"]:
        choice=selection["choices"][str(h)]
        if choice["variant"] not in cfg["variants"] or choice[weight_key] not in cfg[weight_key+"_grid"]:
            raise ValueError("Selected coefficient outside declared experiment")
        # Inspect saved cloud metrics, without recomputing a search or replacing coefficients.
        chosen=grid[(grid.horizon==h)&(grid.variant==choice["variant"])&(grid[weight_key]==choice[weight_key])].iloc[0]
        for row in grid[grid.horizon==h].itertuples():
            if row.mae < chosen.mae or (row.mae==chosen.mae and (getattr(row,weight_key),cfg["variants"].index(row.variant)) < (chosen[weight_key],cfg["variants"].index(choice["variant"]))):
                raise ValueError("Stored choice contradicts saved cloud grid/tie rule")
        fold=cfg["confirmation_fold"]
        with np.load(root/f"confirmation_h{h}.npz",allow_pickle=False) as saved, np.load(linear_root/fold/choice["variant"]/f"score_h{h}.npz",allow_pickle=False) as linear, np.load(hurdle_root/fold/f"h{h}"/"score.npz",allow_pickle=False) as hurdle:
            for key in ("actual","origins","beam_ids"):
                np.testing.assert_array_equal(saved[key],linear[key])
                np.testing.assert_array_equal(saved[key],hurdle[key])
            np.testing.assert_array_equal(saved[component],linear["predicted"])
            np.testing.assert_array_equal(saved["hurdle"],hurdle["hurdle_median"])
            prediction=blend_forecasts(saved["hurdle"],saved[component],choice[weight_key])
            np.testing.assert_array_equal(prediction,saved["predicted"])
            checked+=len(prediction)
            max_error=max(max_error,float(np.max(np.abs(prediction-saved["predicted"]))))
            actual=saved["actual"]
            mask=np.isfinite(actual)
            mae=float(np.abs(prediction[mask]-actual[mask]).mean())
            base=float(np.abs(saved["hurdle"][mask]-actual[mask]).mean())
            rows.append({"fold":fold,"horizon":h,**choice,"mae":mae,
                         "wape":float(np.abs(prediction[mask]-actual[mask]).sum()/np.abs(actual[mask]).sum()),
                         "n_valid":int(mask.sum()),"hurdle_mae":base,"relative_mae_gain_pct":100*(1-mae/base)})
    frame=pd.DataFrame(rows)
    saved_frame=pd.read_csv(f"reports/{report_prefix}_confirmation.csv")
    columns=["fold","horizon","variant",weight_key,"mae","wape","n_valid","hurdle_mae","relative_mae_gain_pct"]
    pd.testing.assert_frame_equal(frame[columns],saved_frame[columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    passes=bool((frame.mae<=frame.hurdle_mae).all() and ((frame[weight_key]>0)&(frame.mae<frame.hurdle_mae)).any())
    decision=json.loads((root/"decision.json").read_text())
    if decision["passes_declared_development_confirmation"]!=passes or decision["holdout_used"]:
        raise ValueError("Stored decision contradicts fixed confirmation")
    Path(f"reports/{report_prefix}_verification.json").write_text(json.dumps({
        "artifact_files_verified":file_count,"cloud_choice_consistency_checked":True,
        "confirmation_predictions_recomputed":checked,"max_prediction_error":max_error,
        "all_confirmation_metrics_reconstructed":True,"passes_declared_development_confirmation":passes,
        "weights_changed_locally":False,"validation_504_671_used":False,"holdout_used":False,
        "claim_scope":"reused development confirmation only, no generalization or promotion claim"},indent=2))
    print(frame.to_string(index=False))
    print("Passes declared confirmation:",passes)


def main():
    verify_blend()


if __name__=="__main__":
    main()
