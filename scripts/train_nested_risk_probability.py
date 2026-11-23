"""Hosted conditional critical classifier with frozen warning probability models."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.optimize_models import fold_arrays
from scripts.train_hurdle_research import fit_component
from scripts.train_risk_probability import labels_for
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.risk.nested_probability import nested_critical_probability
from src.risk.probability_metrics import probability_metrics, nested_probability_audit
from src.risk.thresholds import Thresholds
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    args=parser.parse_args()
    cfg=load_config("configs/nested_risk_probability.yaml")
    base=load_config(cfg["base_protocol"])
    source=args.source_root
    if sha256(source/"manifest.json")!=cfg["source_manifest_sha256"]:
        raise ValueError("Pinned probability source manifest differs")
    for name,digest in json.loads((source/"manifest.json").read_text())["files"].items():
        if sha256(source/name)!=digest:
            raise ValueError("Source checkpoint checksum mismatch")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    panel=replace(panel,values=panel.values[:504],time=panel.time[:504])
    root=Path("artifacts/nested_risk_probability")
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Refusing to overwrite nested probability artifacts")
    Path("reports").mkdir(exist_ok=True)
    rows=[]
    for fold in base["folds"]:
        if not 167<fold["fit_end"]<fold["stop_end"]<fold["score_end"]<=503:
            raise ValueError("Development-only chronology required")
        thresholds=Thresholds.load(source/fold["name"]/"thresholds.npz")
        if thresholds.fit_end!=fold["fit_end"] or thresholds.beam_ids!=panel.beam_ids:
            raise ValueError("Saved fold threshold chronology/identity differs")
        for h in base["horizons"]:
            old=source/fold["name"]/f"h{h}"
            for level in ("warning","critical"):
                meta=json.loads((old/f"{level}_meta.json").read_text())
                if meta["protocol"]!=base or meta["fold"]!=fold or meta["holdout_used"] or meta["validation_504_671_used"]:
                    raise ValueError("Pinned source development protocol differs")
            _,_,origins,(x,y),(sx,sy),(qx,qy)=fold_arrays(panel,fold,h,base["stride"])
            warning_fit=labels_for(y,thresholds.warning,thresholds.valid)
            warning_stop=labels_for(sy,thresholds.warning,thresholds.valid)
            critical_fit=labels_for(y,thresholds.critical,thresholds.valid)
            critical_stop=labels_for(sy,thresholds.critical,thresholds.valid)
            # Future warning labels filter conditional fit/stopping populations only;
            # they never enter model features or prediction-time routing.
            conditional_fit=np.where(warning_fit==1,critical_fit,np.nan)
            conditional_stop=np.where(warning_stop==1,critical_stop,np.nan)
            if len(np.unique(conditional_fit[np.isfinite(conditional_fit)]))!=2:
                raise ValueError("Both conditional classes must exist")
            model=fit_component(x,conditional_fit,sx,conditional_stop,base,"binary")
            directory=root/fold["name"]/f"h{h}"
            directory.mkdir(parents=True)
            model.save_model(str(directory/"critical_given_warning.txt"))
            conditional=lgb.Booster(model_file=str(directory/"critical_given_warning.txt")).predict(qx,num_threads=base["threads"])
            with np.load(old/"warning_score.npz",allow_pickle=False) as warning, np.load(old/"critical_score.npz",allow_pickle=False) as critical:
                for record in (warning,critical):
                    for key,value in (("actual",qy),("origins",origins),("beam_ids",panel.beam_ids)):
                        np.testing.assert_array_equal(record[key],value)
                probability=nested_critical_probability(warning["predicted"],conditional)
                labels=labels_for(qy,thresholds.critical,thresholds.valid)
                np.testing.assert_array_equal(labels,critical["labels"])
                coherence=nested_probability_audit(warning["predicted"],probability,np.isfinite(labels))
                if coherence["nesting_violations"]:
                    raise ValueError("Conditional architecture must preserve nested probabilities")
                for name,p in (("independent_critical",critical["predicted"]),("nested_critical",probability)):
                    rows.append({"fold":fold["name"],"horizon":h,"model":name,**probability_metrics(labels,p)})
                np.savez_compressed(directory/"score.npz",actual=qy,labels=labels,origins=origins,
                    beam_ids=np.asarray(panel.beam_ids),warning=warning["predicted"],conditional=conditional,
                    independent=critical["predicted"],nested=probability)
            (directory/"meta.json").write_text(json.dumps({"protocol":cfg,"base_protocol":base,"fold":fold,
                "horizon":h,"threshold_fit_end":thresholds.fit_end,"best_iteration":model.best_iteration,
                "fit_conditional_count":int(np.isfinite(conditional_fit).sum()),"stop_conditional_count":int(np.isfinite(conditional_stop).sum()),
                "source_commit":os.environ.get("SOURCE_GIT_COMMIT"),"holdout_used":False,"validation_504_671_used":False,
                "nesting_audit":coherence},indent=2))
            pd.DataFrame(rows).to_csv("reports/nested_risk_probability_development.csv",index=False)
            print(f"Completed {fold['name']} h{h}: {rows[-2:]}",flush=True)
    (root/"manifest.json").write_text(json.dumps({"files":{p.relative_to(root).as_posix():sha256(p)
        for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}},indent=2))


if __name__=="__main__":
    main()
