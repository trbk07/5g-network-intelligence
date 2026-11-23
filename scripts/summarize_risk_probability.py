"""Reconstruct probability study labels/metrics and history-only predictions; no fitting."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.train_risk_probability import labels_for
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.risk.probability_metrics import probability_metrics
from src.risk.thresholds import Thresholds
from src.utils.config import load_config


def verify_threshold_statistics(panel,thresholds,fold,policy):
    """Compare saved thresholds with fit-only statistics; never replace them."""
    if thresholds.fit_end != fold["fit_end"] or thresholds.beam_ids != panel.beam_ids:
        raise ValueError("Threshold fit prefix or beam identity differs")
    values=panel.values[:fold["fit_end"]+1,:,0]
    counts=np.isfinite(values).sum(0)
    active=np.divide(((values>policy["epsilon"])&np.isfinite(values)).sum(0),counts,
                     out=np.zeros(len(counts)),where=counts>0)
    np.testing.assert_array_equal(thresholds.n_observed,counts)
    np.testing.assert_array_equal(thresholds.active_ratio,active)
    reasons=[]
    for b in range(len(panel.beam_ids)):
        observed=values[np.isfinite(values[:,b]),b]
        quantiles=np.quantile(observed,policy["quantiles"],method="linear") if len(observed) else [np.nan,np.nan]
        np.testing.assert_allclose([thresholds.warning[b],thresholds.critical[b]],quantiles,atol=0,rtol=0,equal_nan=True)
        reason=("insufficient_observations" if counts[b]<policy["min_observations"] else
                "near_zero_threshold" if quantiles[0]<=policy["epsilon"] else
                "low_activity" if active[b]<policy["min_active_ratio"] else "eligible")
        reasons.append(reason)
    np.testing.assert_array_equal(thresholds.reasons,reasons)
    np.testing.assert_array_equal(thresholds.valid,(np.asarray(reasons)=="eligible")&np.isfinite(thresholds.warning)&np.isfinite(thresholds.critical))


def main():
    cfg=load_config("configs/risk_probability.yaml")
    policy=load_config("configs/risk.yaml")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    root=Path("artifacts/risk_probability")
    files=json.loads((root/"manifest.json").read_text())["files"]
    for name,digest in files.items():
        if sha256(root/name)!=digest:
            raise ValueError("Probability artifact checksum differs")
    rows,parity,comparisons=[],[],[]
    for fold in cfg["folds"]:
        if not 167<fold["fit_end"]<fold["stop_end"]<fold["score_end"]<=503:
            raise ValueError("Development-only chronology required")
        thresholds=Thresholds.load(root/fold["name"]/"thresholds.npz")
        verify_threshold_statistics(panel,thresholds,fold,policy)
        for h in cfg["horizons"]:
            origins=evaluation_origins(fold["stop_end"],fold["score_end"],h)
            train=train_origins(fold["fit_end"],h,stride=cfg["stride"])
            actual=panel.values[origins+h,:,0].reshape(-1)
            for level in cfg["thresholds"]:
                directory=root/fold["name"]/f"h{h}"
                meta=json.loads((directory/f"{level}_meta.json").read_text())
                if (meta["protocol"]!=cfg or meta["risk_policy"]!=policy or meta["fold"]!=fold
                        or meta["horizon"]!=h or meta["threshold"]!=level or meta["threshold_fit_end"]!=fold["fit_end"]
                        or meta["feature_names"]!=FEATURE_NAMES or meta["holdout_used"] or meta["validation_504_671_used"]):
                    raise ValueError("Downloaded probability protocol differs")
                model=lgb.Booster(model_file=str(directory/f"{level}.txt"))
                if (model.feature_name()!=FEATURE_NAMES or model.num_feature()!=40
                        or model.current_iteration()!=meta["best_iteration"]
                        or not 0<model.current_iteration()<=cfg["max_rounds"]
                        or model.params["objective"]!="binary"):
                    raise ValueError("Classifier schema/objective/rounds differ")
                cutoff=getattr(thresholds,level)
                labels=labels_for(actual,cutoff,thresholds.valid)
                fitting_labels=labels_for(panel.values[train+h,:,0],cutoff,thresholds.valid)
                prior=float(np.nanmean(fitting_labels))
                np.testing.assert_equal(meta["fit_prevalence"],prior)
                with np.load(directory/f"{level}_score.npz",allow_pickle=False) as score:
                    for key,value in (("actual",actual),("labels",labels),("origins",origins),("beam_ids",panel.beam_ids)):
                        np.testing.assert_array_equal(score[key],value)
                    pair=[]
                    for name,p in (("direct_probability",score["predicted"]),("fit_prevalence",np.full(len(labels),prior))):
                        row={"fold":fold["name"],"horizon":h,"threshold":level,"model":name,**probability_metrics(labels,p)}
                        rows.append(row)
                        pair.append(row)
                    comparisons.append({"fold":fold["name"],"horizon":h,"threshold":level,
                        "direct_brier":pair[0]["brier"],"prior_brier":pair[1]["brier"],
                        "brier_skill_vs_fit_prior":1-pair[0]["brier"]/pair[1]["brier"],
                        "direct_AP":pair[0]["average_precision"],"scoring_base_rate":pair[0]["base_rate"],
                        "recall_at_0.5":pair[0]["recall"],"f1_at_0.5":pair[0]["f1"]})
                    beams=np.array([0,len(panel.beam_ids)//2,len(panel.beam_ids)-1])
                    for index in (0,len(origins)-1):
                        origin=int(origins[index])
                        x=build_features(panel.values[origin-167:origin+1,beams,:],np.array([167]),h,phase_origins=np.array([origin]))
                        p=model.predict(x,num_threads=4)
                        saved=score["predicted"].reshape(len(origins),len(panel.beam_ids))[index,beams]
                        np.testing.assert_allclose(p,saved,atol=1e-10,rtol=0)
                        parity.append({"fold":fold["name"],"horizon":h,"threshold":level,"origin":origin,
                            "beams_checked":len(beams),"max_error":float(np.max(np.abs(p-saved)))})
    frame=pd.DataFrame(rows)
    declared=pd.read_csv("reports/risk_probability_development.csv",dtype={"flags":"string"}).fillna({"flags":""})
    pd.testing.assert_frame_equal(frame,declared[frame.columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    prefix="reports/risk_probability"
    pd.DataFrame(comparisons).to_csv(prefix+"_comparison.csv",index=False)
    pd.DataFrame(parity).to_csv(prefix+"_cpu_parity.csv",index=False)
    Path(prefix+"_verification.json").write_text(json.dumps({"artifact_files_verified":len(files),
        "fold_fit_threshold_statistics_verified":True,"all_labels_axes_and_metrics_reconstructed":True,
        "all_classifiers_use_original_40_inputs":True,"history_only_predictions_checked":sum(p["beams_checked"] for p in parity),
        "max_parity_error":max(p["max_error"] for p in parity),"thresholds_refitted_or_changed_locally":False,
        "decision_threshold_selected":False,"validation_or_holdout_used":False,"production_promoted":False,
        "claim_scope":"reused development probability study, not independent generalization or operational congestion"},indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
