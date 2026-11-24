"""Verify coherent probability study artifacts and fixed metrics, without fitting."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.summarize_risk_probability import verify_threshold_statistics
from scripts.train_risk_probability import labels_for
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.risk.nested_probability import nested_critical_probability
from src.risk.probability_metrics import probability_metrics, nested_probability_audit
from src.risk.thresholds import Thresholds
from src.utils.config import load_config


def main():
    cfg=load_config("configs/nested_risk_probability.yaml")
    base=load_config(cfg["base_protocol"])
    policy=load_config("configs/risk.yaml")
    source,root=Path("artifacts/risk_probability"),Path("artifacts/nested_risk_probability")
    if sha256(source/"manifest.json")!=cfg["source_manifest_sha256"]:
        raise ValueError("Pinned independent probability source differs")
    for folder in (source,root):
        files=json.loads((folder/"manifest.json").read_text())["files"]
        for name,digest in files.items():
            if sha256(folder/name)!=digest:
                raise ValueError("Probability checkpoint checksum mismatch")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    rows,comparisons,parity,coherence=[],[],[],[]
    for fold in base["folds"]:
        if not 167<fold["fit_end"]<fold["stop_end"]<fold["score_end"]<=503:
            raise ValueError("Development chronology required")
        thresholds=Thresholds.load(source/fold["name"]/"thresholds.npz")
        verify_threshold_statistics(panel,thresholds,fold,policy)
        for h in base["horizons"]:
            old=source/fold["name"]/f"h{h}"
            directory=root/fold["name"]/f"h{h}"
            meta=json.loads((directory/"meta.json").read_text())
            if (meta["protocol"]!=cfg or meta["base_protocol"]!=base or meta["fold"]!=fold
                    or meta["horizon"]!=h or meta["threshold_fit_end"]!=fold["fit_end"]
                    or meta["holdout_used"] or meta["validation_504_671_used"]):
                raise ValueError("Conditional probability protocol differs")
            for key,origins in (("fit",train_origins(fold["fit_end"],h,stride=base["stride"])),
                                ("stop",evaluation_origins(fold["fit_end"],fold["stop_end"],h))):
                warning_labels=labels_for(panel.values[origins+h,:,0],thresholds.warning,thresholds.valid)
                if int((warning_labels==1).sum())!=meta[f"{key}_conditional_count"]:
                    raise ValueError("Conditional population count differs")
            model=lgb.Booster(model_file=str(directory/"critical_given_warning.txt"))
            warning_model=lgb.Booster(model_file=str(old/"warning.txt"))
            if (model.feature_name()!=FEATURE_NAMES or model.num_feature()!=40
                    or model.current_iteration()!=meta["best_iteration"]
                    or not 0<model.current_iteration()<=base["max_rounds"] or model.params["objective"]!="binary"):
                raise ValueError("Conditional classifier schema/objective/rounds differ")
            origins=evaluation_origins(fold["stop_end"],fold["score_end"],h)
            actual=panel.values[origins+h,:,0].reshape(-1)
            labels=labels_for(actual,thresholds.critical,thresholds.valid)
            with np.load(directory/"score.npz",allow_pickle=False) as score, np.load(old/"warning_score.npz",allow_pickle=False) as warning, np.load(old/"critical_score.npz",allow_pickle=False) as independent:
                for record in (score,warning,independent):
                    for key,value in (("actual",actual),("origins",origins),("beam_ids",panel.beam_ids)):
                        np.testing.assert_array_equal(record[key],value)
                np.testing.assert_array_equal(score["labels"],labels)
                np.testing.assert_array_equal(score["labels"],independent["labels"])
                np.testing.assert_array_equal(score["warning"],warning["predicted"])
                np.testing.assert_array_equal(score["independent"],independent["predicted"])
                nested=nested_critical_probability(score["warning"],score["conditional"])
                np.testing.assert_array_equal(nested,score["nested"])
                audit=nested_probability_audit(score["warning"],nested,np.isfinite(labels))
                if audit!=meta["nesting_audit"] or audit["nesting_violations"]:
                    raise ValueError("Coherent architecture violates its nesting contract")
                coherence.append({"fold":fold["name"],"horizon":h,**audit})
                pair=[]
                for name,p in (("independent_critical",score["independent"]),("nested_critical",nested)):
                    row={"fold":fold["name"],"horizon":h,"model":name,**probability_metrics(labels,p)}
                    rows.append(row)
                    pair.append(row)
                comparisons.append({"fold":fold["name"],"horizon":h,
                    "independent_brier":pair[0]["brier"],"nested_brier":pair[1]["brier"],
                    "relative_brier_gain_pct":100*(1-pair[1]["brier"]/pair[0]["brier"]),
                    "independent_AP":pair[0]["average_precision"],"nested_AP":pair[1]["average_precision"],
                    "independent_recall_0.5":pair[0]["recall"],"nested_recall_0.5":pair[1]["recall"]})
                beams=np.array([0,len(panel.beam_ids)//2,len(panel.beam_ids)-1])
                for index in (0,len(origins)-1):
                    origin=int(origins[index])
                    x=build_features(panel.values[origin-167:origin+1,beams,:],np.array([167]),h,phase_origins=np.array([origin]))
                    conditional=model.predict(x,num_threads=4)
                    warning_p=warning_model.predict(x,num_threads=4)
                    for name,p in (("warning",warning_p),("conditional",conditional),
                                   ("nested",nested_critical_probability(warning_p,conditional))):
                        saved=score[name].reshape(len(origins),len(panel.beam_ids))[index,beams]
                        np.testing.assert_allclose(p,saved,atol=1e-10,rtol=0)
                        parity.append({"fold":fold["name"],"horizon":h,"origin":origin,"component":name,
                            "beams_checked":len(beams),"max_error":float(np.max(np.abs(p-saved)))})
    frame=pd.DataFrame(rows)
    saved=pd.read_csv("reports/nested_risk_probability_development.csv",dtype={"flags":"string"}).fillna({"flags":""})
    pd.testing.assert_frame_equal(frame,saved[frame.columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    prefix="reports/nested_risk_probability"
    pd.DataFrame(comparisons).to_csv(prefix+"_comparison.csv",index=False)
    pd.DataFrame(parity).to_csv(prefix+"_cpu_parity.csv",index=False)
    pd.DataFrame(coherence).to_csv(prefix+"_coherence.csv",index=False)
    Path(prefix+"_verification.json").write_text(json.dumps({"artifact_files_verified":len(files),
        "threshold_statistics_and_conditional_population_counts_verified":True,"all_metrics_and_axes_reconstructed":True,
        "warning_probabilities_unchanged":True,"full_chain_rule_arithmetic_verified":True,
        "nested_event_violations":0,"history_only_predictions_checked":sum(p["beams_checked"] for p in parity),
        "max_parity_error":max(p["max_error"] for p in parity),"validation_or_holdout_used":False,
        "local_fit_or_calibration_performed":False,"production_promoted":False,
        "claim_scope":"one-seed reused development probability study, not original-target MAE/WAPE improvement"},indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=="__main__":
    main()
