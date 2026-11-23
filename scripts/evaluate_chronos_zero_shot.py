"""Hosted GPU zero-shot development forecasts, with no fitting or calibration."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.zero_shot_context import target_context
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()  # This exploratory job is intentionally hosted-only.
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA GPU required; no local inference fallback")
    from chronos import ChronosBoltPipeline

    cfg=load_config("configs/chronos_zero_shot.yaml")
    if (cfg["context_length"]!=168 or cfg["prediction_length"]!=24 or cfg["horizons"]!=[1,6,24]
            or cfg["quantile"]!=0.5 or any(fold["score_end"]>503 for fold in cfg["folds"])):
        raise ValueError("Fixed development-only median protocol required")
    for name,expected in (("chronos-forecasting",cfg["chronos_version"]),("transformers",cfg["transformers_version"])):
        if importlib.metadata.version(name)!=expected:
            raise ValueError(f"Dependency version differs: {name}")
    torch.set_num_threads(2)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=Path(snapshot_download(cfg["model_id"],revision=cfg["revision"],
        allow_patterns=["config.json","model.safetensors","README.md","LICENSE"],local_dir="pretrained_checkpoint"))
    if sha256(checkpoint/"model.safetensors")!=cfg["weights_sha256"]:
        raise ValueError("Pinned external checkpoint checksum differs")
    pipeline=ChronosBoltPipeline.from_pretrained(str(checkpoint),device_map="cuda",torch_dtype=torch.float32,
                                                trust_remote_code=False,use_safetensors=True)
    pipeline.model.eval()
    pipeline.model.requires_grad_(False)
    if 0.5 not in pipeline.quantiles:
        raise ValueError("Native median quantile required; do not tune/interpolate a new point estimator")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    values=panel.values[:504]
    root=Path("artifacts/chronos_zero_shot")
    root.mkdir(parents=True,exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    identity={"protocol":cfg,"panel_sha256":sha256(Path("data/processed/panel.npz")),
              "source_commit":os.environ.get("SOURCE_GIT_COMMIT"),"torch_version":str(torch.__version__),
              "gpu":torch.cuda.get_device_name(0),"fine_tuning_performed":False,"holdout_used":False}
    identity_path=root/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("Existing forecast cache belongs to a different run")
    identity_path.write_text(json.dumps(identity,indent=2))
    rows=[]
    for fold in cfg["folds"]:
        directory=root/fold["name"]
        directory.mkdir(exist_ok=True)
        all_origins=evaluation_origins(fold["history_boundary"],fold["score_end"],1)
        predictions=[]
        for index,origin in enumerate(all_origins):
            started=time.perf_counter()
            cache=directory/f"origin_{origin}.npz"
            if cache.exists():
                with np.load(cache,allow_pickle=False) as saved:
                    forecast=saved["predicted"]
            else:
                batches=[]
                for start in range(0,len(panel.beam_ids),cfg["batch_size"]):
                    beams=np.arange(start,min(start+cfg["batch_size"],len(panel.beam_ids)))
                    context=torch.as_tensor(target_context(values,int(origin),beams))
                    with torch.inference_mode():
                        quantiles,_=pipeline.predict_quantiles(inputs=context,prediction_length=24,
                            quantile_levels=[0.5],limit_prediction_length=True)
                    if tuple(quantiles.shape)!=(len(beams),24,1):
                        raise ValueError("Unexpected checkpoint forecast tensor shape")
                    point=quantiles.detach().cpu().numpy()[:,:,0]
                    batches.append(np.maximum(0,point[:,[h-1 for h in cfg["horizons"]]]))
                forecast=np.concatenate(batches)
                np.savez_compressed(cache,predicted=forecast)
            if forecast.shape!=(len(panel.beam_ids),3) or not np.isfinite(forecast).all():
                raise ValueError("Every original beam needs all three finite forecasts")
            predictions.append(forecast)
            print(f"{fold['name']} origin {index+1}/{len(all_origins)} completed in {time.perf_counter()-started:.2f}s",flush=True)
        predictions=np.stack(predictions)
        for column,h in enumerate(cfg["horizons"]):
            origins=evaluation_origins(fold["history_boundary"],fold["score_end"],h)
            prediction=predictions[:len(origins),:,column]
            actual=values[origins+h,:,0]
            observed=np.isfinite(actual)
            error=np.abs(prediction[observed]-actual[observed])
            zero=observed&(actual==0)
            rows.append({"fold":fold["name"],"horizon":h,"model":"chronos_bolt_base_median",
                "mae":float(error.mean()),"wape":float(error.sum()/np.abs(actual[observed]).sum()),
                "zero_fp_rate":float((prediction[zero]>0).mean()),"n_valid":int(observed.sum())})
            np.savez_compressed(directory/f"score_h{h}.npz",predicted=prediction,actual=actual,
                                origins=origins,beam_ids=np.asarray(panel.beam_ids))
        pd.DataFrame(rows).to_csv("reports/chronos_zero_shot_development.csv",index=False)
    if sha256(checkpoint/"model.safetensors")!=cfg["weights_sha256"] or any(p.requires_grad for p in pipeline.model.parameters()):
        raise ValueError("Frozen checkpoint invariant failed")
    files={p.relative_to(root).as_posix():sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files":files},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
