"""Hosted-only nonlinear history study; independent stopping/scoring windows."""

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_temporal_linear import score_horizon
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins,train_origins_joint
from src.evaluation.validation import validation_inputs
from src.features.gru_inputs import GRUBatches,GRUPreprocessor
from src.models.gru_train import train_gru
from src.models.temporal_mlp import TemporalResidualMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed


def train_study(config_path="configs/temporal_mlp.yaml",artifact_path="artifacts/temporal_mlp",
                report_prefix="temporal_mlp",architecture="temporal-residual-mlp-v1",model_factory=None,
                model_initializer=None,training_loss=None):
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Temporal neural study requires hosted CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(2)
    cfg=load_config(config_path)
    if inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default!=cfg["weight_decay"]:
        raise ValueError("Shared optimizer default differs from fixed protocol")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    panel=replace(panel,values=panel.values[:504],time=panel.time[:504])
    root=Path(artifact_path)
    root.mkdir(parents=True,exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    rows=[]
    for fold in cfg["folds"]:
        if not 167<fold["fit_end"]<fold["stop_end"]<fold["score_end"]<=503:
            raise ValueError("Development boundaries required")
        for variant in cfg["variants"]:
            set_seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.use_deterministic_algorithms(True)
            preprocessing=GRUPreprocessor.fit(panel,fold["fit_end"],variant)
            beams=np.arange(len(panel.beam_ids))
            train=GRUBatches(panel,preprocessing,variant,train_origins_joint(fold["fit_end"],stride=cfg["training_stride"]),beams,"cuda")
            stopping=GRUBatches(panel,preprocessing,variant,evaluation_origins(fold["fit_end"],fold["stop_end"],24),beams,"cuda")
            directory=root/fold["name"]/variant
            directory.mkdir(parents=True,exist_ok=True)
            if (directory/"last.pt").exists():
                raise ValueError("Refusing to overwrite an existing run")
            preprocessing.save(directory/"preprocessor.npz")
            model=(model_factory(cfg,variant) if model_factory is not None else
                   TemporalResidualMLP(1 if variant=="A" else 4,cfg["hidden_size"],cfg["dropout"]))
            if model_initializer is not None:
                model=model_initializer(model,panel,preprocessing,fold,cfg)
            identity={"architecture":architecture,"fold":fold,"variant":variant}
            summary=train_gru(model,train,stopping,cfg,directory,cfg["seed"],run_identity=identity,
                              training_loss=training_loss)
            checkpoint=torch.load(directory/"best.pt",map_location="cuda",weights_only=True)
            model.load_state_dict(checkpoint["model"])
            for h in cfg["horizons"]:
                origins,prediction,actual=score_horizon(model,panel,preprocessing,fold,h,cfg["batch_size"])
                observed=np.isfinite(actual)
                if not np.isfinite(prediction[observed]).all():
                    raise ValueError("Missing forecasts cannot shrink evaluation coverage")
                error=np.abs(prediction[observed]-actual[observed])
                zero=observed&(actual==0)
                rows.append({"fold":fold["name"],"variant":variant,"horizon":h,
                    "mae":float(error.mean()),"wape":float(error.sum()/np.abs(actual[observed]).sum()),
                    "zero_fp_rate":float((prediction[zero]>0).mean()),"n_valid":int(observed.sum())})
                np.savez_compressed(directory/f"score_h{h}.npz",actual=actual,predicted=prediction,
                                    origins=origins,beam_ids=np.asarray(panel.beam_ids))
            (directory/"meta.json").write_text(json.dumps({"protocol":cfg,"fold":fold,"variant":variant,
                "training":summary,"parameters":sum(p.numel() for p in model.parameters()),
                "source_commit":os.environ.get("SOURCE_GIT_COMMIT"),"torch_version":str(torch.__version__),
                "gpu":torch.cuda.get_device_name(0),"holdout_used":False,"validation_504_671_used":False},indent=2))
            pd.DataFrame(rows).to_csv(f"reports/{report_prefix}_development.csv",index=False)
            print(pd.DataFrame(rows).tail(3).to_string(index=False),flush=True)
            del model,train,stopping
            torch.cuda.empty_cache()
    files={p.relative_to(root).as_posix():sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files":files},indent=2))


def main():
    train_study()


if __name__=="__main__":
    main()
