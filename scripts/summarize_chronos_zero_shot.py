"""Check saved origin-cache mapping and metrics without loading pretrained weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_hurdle_research import score_rows
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.utils.config import load_config


def main():
    cfg=load_config("configs/chronos_zero_shot.yaml")
    root=Path("artifacts/chronos_zero_shot")
    manifest=json.loads((root/"manifest.json").read_text())
    for name,digest in manifest["files"].items():
        if sha256(root/name)!=digest:
            raise ValueError("Zero-shot artifact checksum mismatch")
    identity=json.loads((root/"identity.json").read_text())
    if (identity["protocol"]!=cfg or identity["fine_tuning_performed"] or identity["holdout_used"]
            or identity["panel_sha256"]!=sha256(Path("data/processed/panel.npz"))):
        raise ValueError("Pinned zero-shot run identity differs")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    rows,comparisons=[],[]
    predictions_checked=0
    for fold in cfg["folds"]:
        if fold["score_end"]>503:
            raise ValueError("Development target boundary exceeds 503")
        all_origins=evaluation_origins(fold["history_boundary"],fold["score_end"],1)
        cached=[]
        for origin in all_origins:
            with np.load(root/fold["name"]/f"origin_{origin}.npz",allow_pickle=False) as cache:
                value=cache["predicted"]
                if value.shape!=(len(panel.beam_ids),3) or not np.isfinite(value).all() or (value<0).any():
                    raise ValueError("Complete nonnegative original-beam cache required")
                cached.append(value)
        cached=np.stack(cached)
        for column,h in enumerate(cfg["horizons"]):
            origins=evaluation_origins(fold["history_boundary"],fold["score_end"],h)
            with np.load(root/fold["name"]/f"score_h{h}.npz",allow_pickle=False) as score:
                actual=panel.values[origins+h,:,0]
                np.testing.assert_array_equal(score["origins"],origins)
                np.testing.assert_array_equal(score["beam_ids"],panel.beam_ids)
                np.testing.assert_array_equal(score["actual"],actual)
                np.testing.assert_array_equal(score["predicted"],cached[:len(origins),:,column])
                predictions_checked+=score["predicted"].size
                result=score_rows(actual,{"chronos_bolt_base_median":score["predicted"]},fold["name"],h)[0]
                rows.append(result)
                with np.load(Path("artifacts/hurdle_research")/fold["name"]/f"h{h}"/"score.npz",allow_pickle=False) as reference:
                    np.testing.assert_array_equal(reference["origins"],origins)
                    np.testing.assert_array_equal(reference["actual"].reshape(actual.shape),actual)
                    np.testing.assert_array_equal(reference["beam_ids"],panel.beam_ids)
                    observed=np.isfinite(actual)
                    baseline=float(np.abs(reference["hurdle_median"].reshape(actual.shape)[observed]-actual[observed]).mean())
                comparisons.append({**result,"hurdle_mae":baseline,"relative_mae_gain_vs_hurdle_pct":100*(1-result["mae"]/baseline)})
    frame=pd.DataFrame(rows)
    declared=pd.read_csv("reports/chronos_zero_shot_development.csv")
    columns=["fold","horizon","model","mae","wape","zero_fp_rate","n_valid"]
    pd.testing.assert_frame_equal(frame[columns],declared[columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv("reports/chronos_zero_shot_comparison.csv",index=False)
    Path("reports/chronos_zero_shot_verification.json").write_text(json.dumps({
        "artifact_files_verified":len(manifest["files"]),"score_predictions_checked_against_origin_cache":predictions_checked,
        "all_metrics_reconstructed":True,"origins_targets_and_beams_match_hurdle":True,
        "independent_pretrained_model_execution_verified":False,"weights_downloaded_locally":False,
        "pretraining_overlap_certified_absent":False,"validation_504_671_used":False,"holdout_used":False,
        "claim_scope":"saved-output consistency on reused development only; no certified zero-shot generalization claim"},indent=2))
    print(comparison.to_string(index=False))


if __name__=="__main__":
    main()
