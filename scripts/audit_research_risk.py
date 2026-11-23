"""Evaluate saved research forecasts with unchanged historical thresholds; no fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.risk.evaluator import evaluate_exceedance
from src.risk.thresholds import Thresholds
from src.utils.config import load_config


THRESHOLD_SHA256 = "6794998429479a1838ea78932134e73d7417b78f7806c0fe35fa4d3532ed70c6"
SOURCES = {
    "baselines_val.npz": ("persistence", "seasonal_naive_24"),
    "lightgbm_val.npz": ("lightgbm_c1",),
    "improvement_audit.npz": ("improved_ensemble",),
    "hurdle_candidate_val.npz": ("hurdle_median",),
    "blend_candidate_val.npz": ("hurdle_linear_blend",),
    "mlp_blend_candidate_val.npz": ("hurdle_mlp_blend",),
}


def main():
    threshold_path=Path("reports/thresholds_validation.npz")
    if sha256(threshold_path) != THRESHOLD_SHA256:
        raise ValueError("Frozen validation thresholds changed")
    thresholds=Thresholds.load(threshold_path)
    if thresholds.fit_end != 503:
        raise ValueError("Thresholds must use the original fit prefix only")
    cfg=load_config("configs/risk.yaml")
    rows=[]
    reference=risk_records(Path("reports/forecasts/lightgbm_val.npz"))
    hashes={}
    for filename,names in SOURCES.items():
        path=Path("reports/forecasts")/filename
        hashes[filename]=sha256(path)
        records=risk_records(path)
        if set(records) != {1,6,24}:
            raise ValueError("Every horizon must be retained")
        for h,group in records.items():
            expected=reference[h]["lightgbm_c1"]
            for name in names:
                record=group[name]
                for key in ("actual","origins"):
                    np.testing.assert_array_equal(record[key],expected[key])
                np.testing.assert_array_equal(record["origins"],evaluation_origins(503,671,h))
                if record["beam_ids"] != thresholds.beam_ids or record["beam_ids"] != expected["beam_ids"]:
                    raise ValueError("Full original beam axis required")
                mask=np.isfinite(record["actual"])
                if not np.isfinite(record["predicted"][mask]).all():
                    raise ValueError("Missing forecasts cannot shrink risk scoring coverage")
                for result in evaluate_exceedance(record["actual"],record["predicted"],thresholds,mask,cfg["min_positive_report"]):
                    if sum(result[k] for k in ("TP","FP","FN","TN")) != result["n_valid"]:
                        raise ValueError("Risk confusion counts do not reconcile")
                    rows.append({"segment":"val","model":name,"horizon":h,**result})
    frame=pd.DataFrame(rows)
    # Original baseline/C1 rows must reproduce the existing evaluation exactly.
    original=pd.read_csv("reports/risk_validation.csv",dtype={"flags":"string"}).fillna({"flags":""})
    names=("persistence","seasonal_naive_24","lightgbm_c1")
    keys=["model","horizon","threshold"]
    actual=frame[frame.model.isin(names)].set_index(keys).sort_index()
    expected=original[original.model.isin(names)].set_index(keys).sort_index()
    pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    frame.to_csv("reports/research_risk_validation.csv",index=False)
    Path("reports/research_risk_verification.json").write_text(json.dumps({
        "thresholds_sha256":THRESHOLD_SHA256,"threshold_fit_end":503,"thresholds_refitted":False,
        "policy_retuned":False,"original_baseline_C1_rows_reproduced":True,
        "confusion_counts_reconcile":True,"rows":len(frame),"forecast_sha256":hashes,
        "holdout_used":False,"production_promoted":False,
        "claim_scope":"historical P90/P95 exceedance on reused validation; no operational congestion or future-time guarantee"
    },indent=2))
    print(frame[frame.model.isin(("hurdle_median","hurdle_linear_blend","hurdle_mlp_blend"))][
        ["model","horizon","threshold","TP","FP","FN","precision","recall","f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
