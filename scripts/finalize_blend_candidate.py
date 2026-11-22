"""Hosted fixed-epoch linear refit, frozen blend and reused-validation audit."""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_risk import risk_records
from scripts.train_temporal_linear import score_horizon
from src.data.io import sha256
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import train_origins_joint
from src.evaluation.validation import validation_inputs
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.models.forecast_blend import blend_forecasts
from src.models.gru_train import _save, masked_l1
from src.models.temporal_linear import TemporalLinear
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed


def fixed_epochs(metadata):
    """Arithmetic on two stopping-selected counts; never inspect audit scores."""
    if len(metadata)!=2:
        raise ValueError("Two original development epoch counts required")
    counts=[]
    for meta in metadata:
        training=meta["training"]
        best=training["best_epoch"]
        if (not isinstance(best,int) or best<0 or training["epochs_completed"]<=best
                or training["budget_exhausted"]):
            raise ValueError("Complete stopping-selected development epochs required")
        counts.append(best+1)
    return int(np.floor(np.median(counts)+0.5))


def train_fixed(model, batches, cfg, epochs, directory, identity):
    require_cloud_training()
    if batches.device!="cuda" or epochs<1:
        raise ValueError("Fixed refit requires hosted CUDA batches and positive epochs")
    model.to("cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    for epoch in range(epochs):
        started=time.perf_counter()
        model.train()
        total,count=0.,0
        for sequence,phase,target in batches.batches(cfg["batch_size"],cfg["seed"]+epoch):
            optimizer.zero_grad(set_to_none=True)
            loss,n=masked_l1(model(sequence,phase),target)
            if not n:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"])
            optimizer.step()
            total+=float(loss.detach())*n
            count+=n
        if not count:
            raise ValueError("No observed fit targets")
        _save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
               "identity":identity,"rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},directory/"last.pt")
        row={"epoch":epoch,"train_l1_scaled":total/count,"n_targets":count,"seconds":time.perf_counter()-started}
        with (directory/"history.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps({"variant":identity["variant"],**row}),flush=True)
    model.eval()
    return model


def verify_manifest(root):
    for name,digest in json.loads((root/"manifest.json").read_text())["files"].items():
        if sha256(root/name)!=digest:
            raise ValueError("Source artifact checksum mismatch")


def main():
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted GPU is required for fixed neural refit")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-root",type=Path,required=True)
    parser.add_argument("--blend-root",type=Path,required=True)
    parser.add_argument("--hurdle-replay",type=Path,required=True)
    args=parser.parse_args()
    cfg=load_config("configs/blend_candidate_audit.yaml")
    if cfg["fit_end"]!=503 or cfg["audit_end"]!=671 or cfg["horizons"]!=[1,6,24]:
        raise ValueError("Fixed original development/audit boundaries required")
    base=load_config("configs/temporal_linear.yaml")
    verify_manifest(args.linear_root)
    verify_manifest(args.blend_root)
    if (sha256(args.blend_root/"selection.json")!=cfg["selection_sha256"]
            or sha256(args.hurdle_replay)!=cfg["hurdle_replay_sha256"]):
        raise ValueError("Pinned blend choices or saved hurdle replay differ")
    decision=json.loads((args.blend_root/"decision.json").read_text())
    if not decision["passes_declared_development_confirmation"]:
        raise ValueError("Development confirmation must pass before this separate audit")
    selection=json.loads((args.blend_root/"selection.json").read_text())
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    fit_panel=replace(panel,values=panel.values[:cfg["fit_end"]+1],time=panel.time[:cfg["fit_end"]+1])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root=Path("artifacts/blend_candidate")
    root.mkdir(parents=True,exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    models={}
    variants={choice["variant"] for choice in selection["choices"].values()}
    for variant in base["variants"]:
        if variant not in variants:
            continue
        metadata=[json.loads((args.linear_root/fold["name"]/variant/"meta.json").read_text()) for fold in base["folds"]]
        for meta,fold in zip(metadata,base["folds"],strict=True):
            if meta["protocol"]!=base or meta["fold"]!=fold or meta["variant"]!=variant:
                raise ValueError("Original neural development protocol differs")
        epochs=fixed_epochs(metadata)
        set_seed(base["seed"])
        torch.manual_seed(base["seed"])
        torch.cuda.manual_seed_all(base["seed"])
        preprocessor=GRUPreprocessor.fit(fit_panel,cfg["fit_end"],variant)
        batches=GRUBatches(fit_panel,preprocessor,variant,train_origins_joint(cfg["fit_end"],stride=base["training_stride"]),np.arange(len(panel.beam_ids)),"cuda")
        directory=root/variant
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"last.pt").exists():
            raise ValueError("Refusing to overwrite existing fixed refit")
        identity={"architecture":"temporal-linear-refit-v1","variant":variant,"fit_end":cfg["fit_end"],"epochs":epochs,"seed":base["seed"]}
        preprocessor.save(directory/"preprocessor.npz")
        model=train_fixed(TemporalLinear(1 if variant=="A" else 4),batches,base,epochs,directory,identity)
        (directory/"meta.json").write_text(json.dumps({"identity":identity,"base_protocol":base,"audit_protocol":cfg,
            "source_commit":os.environ.get("SOURCE_GIT_COMMIT"),"development_manifest_sha256":sha256(args.linear_root/"manifest.json"),
            "torch_version":str(torch.__version__),"gpu":torch.cuda.get_device_name(0),
            "holdout_used":False},indent=2))
        models[variant]=(model,preprocessor)
        del batches
    # Build audit forecasts only after every fixed refit is complete.
    reference=risk_records(args.hurdle_replay)
    tiers=assign_load_tiers(fit_panel.values[:,:,0])
    rows,records=[],[]
    for h in cfg["horizons"]:
        choice=selection["choices"][str(h)]
        model,preprocessor=models[choice["variant"]]
        origins,linear,actual=score_horizon(model,panel,preprocessor,{"stop_end":cfg["fit_end"],"score_end":cfg["audit_end"]},h,base["batch_size"])
        saved=reference[h]["hurdle_median"]
        np.testing.assert_array_equal(origins,saved["origins"])
        np.testing.assert_array_equal(actual.reshape(saved["actual"].shape),saved["actual"])
        if panel.beam_ids!=saved["beam_ids"]:
            raise ValueError("Saved hurdle population differs")
        linear=linear.reshape(saved["actual"].shape)
        mixture=blend_forecasts(saved["predicted"],linear,choice["linear_weight"])
        for name,predicted in (("selected_linear",linear),("hurdle_median",saved["predicted"]),("hurdle_linear_blend",mixture)):
            record=ForecastRecord(name,h,origins,panel.beam_ids,saved["actual"].copy(),predicted)
            records.append(record)
            rows.extend(to_benchmark_rows(record,tiers))
        pd.DataFrame(rows).to_csv("reports/blend_candidate_validation.csv",index=False)
    save_forecasts(records,"reports/forecasts/blend_candidate_val.npz",{"research_only":True,"holdout_used":False})
    (root/"selection.json").write_text(json.dumps(selection,indent=2))
    (root/"protocol.json").write_text(json.dumps(cfg,indent=2))
    files={p.relative_to(root).as_posix():sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files":files},indent=2))
    print(pd.DataFrame(rows).query("load_tier == 'All'")[["model","horizon","mae","wape"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
