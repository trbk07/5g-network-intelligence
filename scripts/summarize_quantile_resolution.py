"""Verify fixed finer-quantile forecasts by history-only inference, never fitting."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.summarize_hurdle_research import score_rows
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.hurdle import mixture_median
from src.utils.config import load_config


def main():
    cfg = load_config("configs/hurdle_quantile_resolution.yaml")
    base_cfg = load_config("configs/hurdle_research.yaml")
    source, root = Path("artifacts/hurdle_research"), Path("artifacts/hurdle_quantile_resolution")
    if sha256(source/"manifest.json") != cfg["source_manifest_sha256"]:
        raise ValueError("Original manifest differs")
    for directory in (source,root):
        files = json.loads((directory/"manifest.json").read_text())["files"]
        for name,digest in files.items():
            if sha256(directory/name) != digest:
                raise ValueError("Artifact checksum mismatch")
    panel,_ = validation_inputs(load_config("configs/data.yaml"))
    levels = sorted(cfg["retained_quantiles"]+cfg["additional_quantiles"])
    rows,parity,comparison = [],[],[]
    for fold in base_cfg["folds"]:
        if fold["score_end"] > 503:
            raise ValueError("Development-only chronology required")
        for h in base_cfg["horizons"]:
            directory,old = root/fold["name"]/f"h{h}",source/fold["name"]/f"h{h}"
            meta = json.loads((directory/"meta.json").read_text())
            if (meta["protocol"] != cfg or meta["base_protocol"] != base_cfg or meta["fold"] != fold
                    or meta["horizon"] != h or meta["holdout_used"] or meta["validation_504_671_used"]
                    or set(meta["new_component_rounds"]) != {f"q{a:g}" for a in cfg["additional_quantiles"]}):
                raise ValueError("Downloaded experiment contract differs")
            models = {a:lgb.Booster(model_file=str((old if a in cfg["retained_quantiles"] else directory)/f"q{a:g}.txt")) for a in levels}
            active = lgb.Booster(model_file=str(old/"active.txt"))
            with np.load(directory/"score.npz",allow_pickle=False) as score, np.load(old/"score.npz",allow_pickle=False) as reference:
                origins = evaluation_origins(fold["stop_end"],fold["score_end"],h)
                actual = panel.values[origins+h,:,0].reshape(-1)
                for key,value in (("actual",actual),("origins",origins),("beam_ids",panel.beam_ids)):
                    np.testing.assert_array_equal(score[key],value)
                    np.testing.assert_array_equal(reference[key],value)
                np.testing.assert_allclose(score["original"],reference["hurdle_median"],atol=1e-8,rtol=0)
                pair = score_rows(actual,{name:score[name] for name in ("original","refined")},fold["name"],h)
                rows.extend(pair)
                comparison.append({"fold":fold["name"],"horizon":h,"original_mae":pair[0]["mae"],"refined_mae":pair[1]["mae"],
                    "refined_wape":pair[1]["wape"],"relative_mae_gain_pct":100*(1-pair[1]["mae"]/pair[0]["mae"])})
                beams = np.array([0,len(panel.beam_ids)//2,len(panel.beam_ids)-1])
                for index in (0,len(origins)-1):
                    origin = int(origins[index])
                    history = panel.values[origin-167:origin+1,beams,:]
                    x = build_features(history,np.array([167]),h,phase_origins=np.array([origin]))
                    positive = {a:m.predict(x,num_threads=4) for a,m in models.items()}
                    p = active.predict(x,num_threads=4)
                    for name,grid in (("original",cfg["retained_quantiles"]),("refined",levels)):
                        predicted = mixture_median(p,np.column_stack([positive[a] for a in grid]),grid)
                        saved = score[name].reshape(len(origins),len(panel.beam_ids))[index,beams]
                        np.testing.assert_allclose(predicted,saved,atol=1e-8,rtol=0)
                        parity.append({"fold":fold["name"],"horizon":h,"model":name,"origin":origin,
                                       "beams_checked":len(beams),"max_error":float(np.max(np.abs(predicted-saved)))})
    columns = ["fold","horizon","model","mae","wape","n_valid"]
    pd.testing.assert_frame_equal(pd.DataFrame(rows)[columns],pd.read_csv("reports/hurdle_quantile_resolution_development.csv")[columns],check_dtype=False,atol=1e-12,rtol=1e-12)
    prefix = "reports/hurdle_quantile_resolution"
    pd.DataFrame(comparison).to_csv(prefix+"_comparison.csv",index=False)
    pd.DataFrame(parity).to_csv(prefix+"_cpu_parity.csv",index=False)
    Path(prefix+"_verification.json").write_text(json.dumps({"metrics_reconstructed":True,"all_scoring_axes_verified":True,
        "history_only_predictions_checked":sum(p["beams_checked"] for p in parity),"max_parity_error":max(p["max_error"] for p in parity),
        "tolerance_original_units":1e-8,"validation_or_holdout_used":False,"production_promoted":False,
        "claim_scope":"one-seed reused development evidence only"},indent=2))
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
