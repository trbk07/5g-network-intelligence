"""Hosted controlled positive-quantile refinement; original components stay fixed."""

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
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.models.hurdle import mixture_median
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config("configs/hurdle_quantile_resolution.yaml")
    source = args.source_root
    if sha256(source/"manifest.json") != cfg["source_manifest_sha256"]:
        raise ValueError("Original component manifest differs")
    manifest = json.loads((source/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(source/name) != digest:
            raise ValueError("Original component checksum mismatch")
    base_cfg = load_config("configs/hurdle_research.yaml")
    if base_cfg["conditional_quantiles"] != cfg["retained_quantiles"]:
        raise ValueError("Retained quantile grid differs")
    levels = sorted(cfg["retained_quantiles"]+cfg["additional_quantiles"])
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    root = Path("artifacts/hurdle_quantile_resolution")
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    rows = []
    for fold in base_cfg["folds"]:
        if not fold["fit_end"] < fold["stop_end"] < fold["score_end"] <= 503:
            raise ValueError("Development-only chronology required")
        for h in base_cfg["horizons"]:
            _, _, origins, (x,y), (sx,sy), (qx,qy) = fold_arrays(panel,fold,h,base_cfg["stride"])
            old = source/fold["name"]/f"h{h}"
            meta = json.loads((old/"meta.json").read_text())
            if meta["protocol"] != base_cfg or meta["holdout_used"] or meta["validation_504_671_used"]:
                raise ValueError("Original protocol differs")
            directory = root/fold["name"]/f"h{h}"
            directory.mkdir(parents=True, exist_ok=True)
            positive, rounds = {}, {}
            for a in levels:
                name = f"q{a:g}"
                if a in cfg["retained_quantiles"]:
                    model = lgb.Booster(model_file=str(old/f"{name}.txt"))
                else:
                    model = fit_component(x,np.where(y>0,y,np.nan),sx,np.where(sy>0,sy,np.nan),base_cfg,"quantile",a)
                    model.save_model(str(directory/f"{name}.txt"))
                    rounds[name] = model.best_iteration
                    model = lgb.Booster(model_file=str(directory/f"{name}.txt"))
                positive[a] = model.predict(qx,num_threads=base_cfg["threads"])
            p = lgb.Booster(model_file=str(old/"active.txt")).predict(qx,num_threads=base_cfg["threads"])
            original = mixture_median(p,np.column_stack([positive[a] for a in cfg["retained_quantiles"]]),cfg["retained_quantiles"])
            refined = mixture_median(p,np.column_stack([positive[a] for a in levels]),levels)
            with np.load(old/"score.npz",allow_pickle=False) as score:
                for key,value in (("actual",qy),("origins",origins),("beam_ids",panel.beam_ids)):
                    np.testing.assert_array_equal(score[key],value)
                np.testing.assert_allclose(original,score["hurdle_median"],atol=1e-8,rtol=0)
            mask = np.isfinite(qy)
            for name,prediction in (("original",original),("refined",refined)):
                if not np.isfinite(prediction[mask]).all():
                    raise ValueError("Nonfinite observed-target forecast")
                error = np.abs(prediction[mask]-qy[mask])
                rows.append({"fold":fold["name"],"horizon":h,"model":name,"mae":float(error.mean()),
                             "wape":float(error.sum()/np.abs(qy[mask]).sum()),"n_valid":int(mask.sum())})
            np.savez_compressed(directory/"score.npz",actual=qy,origins=origins,beam_ids=np.asarray(panel.beam_ids),original=original,refined=refined)
            (directory/"meta.json").write_text(json.dumps({"protocol":cfg,"base_protocol":base_cfg,"fold":fold,
                "horizon":h,"new_component_rounds":rounds,"source_commit":os.environ.get("SOURCE_GIT_COMMIT"),
                "holdout_used":False,"validation_504_671_used":False},indent=2))
            pd.DataFrame(rows).to_csv("reports/hurdle_quantile_resolution_development.csv",index=False)
            print(f"Completed {fold['name']} h{h}: {rows[-2:]}",flush=True)
    (root/"manifest.json").write_text(json.dumps({"files":{p.relative_to(root).as_posix():sha256(p)
        for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}},indent=2))


if __name__ == "__main__":
    main()
