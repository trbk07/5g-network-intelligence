"""Hosted GPU-only temporal-linear development study, with saved per-epoch states."""

import json
import inspect
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins, train_origins_joint
from src.evaluation.validation import validation_inputs
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.models.gru_train import train_gru
from src.models.temporal_linear import TemporalLinear
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed


def score_horizon(model, panel, preprocessor, fold, h, batch_size):
    """Score all per-horizon origins, including those lacking a later +24h label."""
    origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
    pairs = np.array(np.meshgrid(origins, np.arange(len(panel.beam_ids)), indexing="ij")).reshape(2, -1).T
    transformed = torch.as_tensor(preprocessor.transform(panel.values, preprocessor.variant), device="cuda")
    offsets = torch.arange(-167, 1, device="cuda")
    head = (1, 6, 24).index(h)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            t, b = torch.as_tensor(chunk[:, 0], device="cuda"), torch.as_tensor(chunk[:, 1], device="cuda")
            sequence = transformed[t[:, None] + offsets, b[:, None]]
            phase = torch.as_tensor(preprocessor.static_at(chunk[:, 0], chunk[:, 1]), device="cuda")
            value = model(sequence, phase)[:, head].cpu().numpy() * preprocessor.scale[0] + preprocessor.mean[0]
            predictions.append(np.maximum(0, value))
    return origins, np.concatenate(predictions), panel.values[origins + h, :, 0].reshape(-1)


def main():
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Temporal-linear fitting requires a hosted CUDA GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    cfg = load_config("configs/temporal_linear.yaml")
    if inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default != cfg["weight_decay"]:
        raise ValueError("Shared trainer AdamW default differs from declared protocol")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    root = Path("artifacts/temporal_linear")
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    rows = []
    for fold in cfg["folds"]:
        if not 167 < fold["fit_end"] < fold["stop_end"] < fold["score_end"] <= 503:
            raise ValueError("Invalid development chronology")
        for variant in cfg["variants"]:
            set_seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.use_deterministic_algorithms(True)
            preprocessor = GRUPreprocessor.fit(panel, fold["fit_end"], variant)
            train = GRUBatches(panel, preprocessor, variant,
                               train_origins_joint(fold["fit_end"], stride=cfg["training_stride"]),
                               np.arange(len(panel.beam_ids)), "cuda")
            stopping = GRUBatches(panel, preprocessor, variant,
                                  evaluation_origins(fold["fit_end"], fold["stop_end"], 24),
                                  np.arange(len(panel.beam_ids)), "cuda")
            directory = root / fold["name"] / variant
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "last.pt").exists():
                raise ValueError("Refusing to overwrite historical execution; use a distinct run")
            preprocessor.save(directory / "preprocessor.npz")
            model = TemporalLinear(1 if variant == "A" else 4)
            summary = train_gru(model, train, stopping, cfg, directory, cfg["seed"],
                                run_identity={"architecture": "temporal-linear-v1", "fold": fold, "variant": variant})
            checkpoint = torch.load(directory / "best.pt", map_location="cuda", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            for h in cfg["horizons"]:
                origins, prediction, actual = score_horizon(model, panel, preprocessor, fold, h, cfg["batch_size"])
                observed = np.isfinite(actual)
                if not np.isfinite(prediction[observed]).all():
                    raise ValueError("Missing predictions cannot reduce scored population")
                errors = np.abs(prediction[observed]-actual[observed])
                zero = observed & (actual == 0)
                rows.append({"fold": fold["name"], "variant": variant, "horizon": h,
                             "mae": float(errors.mean()), "wape": float(errors.sum()/np.abs(actual[observed]).sum()),
                             "zero_fp_rate": float((prediction[zero]>0).mean()), "n_valid": int(observed.sum())})
                np.savez_compressed(directory / f"score_h{h}.npz", predicted=prediction, actual=actual,
                                    origins=origins, beam_ids=np.asarray(panel.beam_ids))
            (directory / "meta.json").write_text(json.dumps({"protocol": cfg, "fold": fold,
                "variant": variant, "training": summary, "parameters": sum(p.numel() for p in model.parameters()),
                "source_commit": os.environ.get("SOURCE_GIT_COMMIT"), "gpu": torch.cuda.get_device_name(0),
                "holdout_used": False, "validation_504_671_used": False}, indent=2))
            pd.DataFrame(rows).to_csv("reports/temporal_linear_development.csv", index=False)
            print(pd.DataFrame(rows).tail(3).to_string(index=False), flush=True)
            del train, stopping, model
            torch.cuda.empty_cache()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
