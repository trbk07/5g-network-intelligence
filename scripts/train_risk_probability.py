"""Hosted direct exceedance probability study; thresholds define labels, not inputs."""

import json
import os
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.optimize_models import fold_arrays
from scripts.train_hurdle_research import fit_component
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.risk.probability_metrics import probability_metrics
from src.risk.thresholds import fit_thresholds
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def labels_for(actual,cutoff,eligible):
    """Map threshold labels without turning missing observations into negatives."""
    actual=np.asarray(actual).reshape(-1,len(cutoff))
    labels=np.where(np.isfinite(actual)&eligible[None,:],actual>=cutoff[None,:],np.nan)
    return labels.reshape(-1)


def main():
    require_cloud_training()
    cfg=load_config("configs/risk_probability.yaml")
    policy=load_config("configs/risk.yaml")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    panel=replace(panel,values=panel.values[:504],time=panel.time[:504])
    root=Path("artifacts/risk_probability")
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Refusing to overwrite a probability study")
    Path("reports").mkdir(exist_ok=True)
    rows=[]
    for fold in cfg["folds"]:
        if not 167<fold["fit_end"]<fold["stop_end"]<fold["score_end"]<=503:
            raise ValueError("Development-only temporal boundaries required")
        prefix=replace(panel,values=panel.values[:fold["fit_end"]+1],time=panel.time[:fold["fit_end"]+1])
        thresholds=fit_thresholds(prefix,policy)
        folder=root/fold["name"]
        folder.mkdir(parents=True)
        thresholds.save(folder/"thresholds.npz")
        for h in cfg["horizons"]:
            _,_,origins,(x,y),(sx,sy),(qx,qy)=fold_arrays(panel,fold,h,cfg["stride"])
            directory=folder/f"h{h}"
            directory.mkdir()
            for level in cfg["thresholds"]:
                cutoff=getattr(thresholds,level)
                fit_y=labels_for(y,cutoff,thresholds.valid)
                stop_y=labels_for(sy,cutoff,thresholds.valid)
                score_y=labels_for(qy,cutoff,thresholds.valid)
                if len(np.unique(fit_y[np.isfinite(fit_y)]))!=2:
                    raise ValueError("Both classes must exist in the fit prefix")
                model=fit_component(x,fit_y,sx,stop_y,cfg,"binary")
                path=directory/f"{level}.txt"
                model.save_model(str(path))
                loaded=lgb.Booster(model_file=str(path))
                probability=loaded.predict(qx,num_threads=cfg["threads"])
                np.testing.assert_allclose(probability[:64],model.predict(qx[:64]),atol=1e-10,rtol=0)
                # The prevalence control is fixed from fitting labels, never scoring labels.
                prior=float(np.nanmean(fit_y))
                for name,p in (("direct_probability",probability),("fit_prevalence",np.full(len(score_y),prior))):
                    rows.append({"fold":fold["name"],"horizon":h,"threshold":level,"model":name,
                                 **probability_metrics(score_y,p)})
                np.savez_compressed(directory/f"{level}_score.npz",actual=qy,labels=score_y,predicted=probability,
                    origins=origins,beam_ids=np.asarray(panel.beam_ids))
                (directory/f"{level}_meta.json").write_text(json.dumps({"protocol":cfg,"risk_policy":policy,
                    "fold":fold,"horizon":h,"threshold":level,"threshold_fit_end":thresholds.fit_end,
                    "fit_prevalence":prior,"best_iteration":model.best_iteration,"source_commit":os.environ.get("SOURCE_GIT_COMMIT"),
                    "feature_names":model.feature_name(),"holdout_used":False,"validation_504_671_used":False},indent=2))
                pd.DataFrame(rows).to_csv("reports/risk_probability_development.csv",index=False)
                print(f"Completed {fold['name']} h{h} {level}: {rows[-2:]}",flush=True)
    (root/"manifest.json").write_text(json.dumps({"files":{p.relative_to(root).as_posix():sha256(p)
        for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}},indent=2))


if __name__ == "__main__":
    main()
