"""Hosted-only matched local/spatial architecture study with durable checkpoints."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_multisource_neural import validate_output_storage
from src.data.io import sha256
from src.evaluation.origins import evaluation_origins, train_origins_joint
from src.evaluation.validation import validation_inputs
from src.features.cell_inputs import CellBatches, cell_groups
from src.features.gru_inputs import GRUPreprocessor
from src.models.cell_attention import CellPatchAttention, configure_attention_math
from src.models.gru_train import train_gru
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_neural_seed


def score(model, panel, preprocessing, fold, directory, device="cuda"):
    groups = cell_groups(panel.beam_ids)
    values = torch.as_tensor(preprocessing.transform(panel.values, "B"), device=device)
    records = {h: {"origins": [], "predicted": [], "actual": []} for h in (1, 6, 24)}
    model.eval()
    with torch.no_grad():
        for origin in range(fold["stop_end"], fold["score_end"]):
            full = np.empty((len(panel.beam_ids), 3), dtype=np.float32)
            for start in range(0, len(groups), 32):
                beams = torch.as_tensor(groups[start:start+32], device=device)
                times = torch.arange(origin-167, origin+1, device=device)
                history = values[times[None, :, None], beams[:, None, :]]
                phase = torch.as_tensor(preprocessing.static_at(np.full(len(beams), origin),
                                        np.arange(len(beams))), device=device)
                full[groups[start:start+32]] = model(history, phase).cpu().numpy()
            for j, h in enumerate((1, 6, 24)):
                if origin+h <= fold["score_end"]:
                    records[h]["origins"].append(origin)
                    records[h]["predicted"].append(full[:, j])
                    records[h]["actual"].append(panel.values[origin+h, :, 0])
    rows = []
    for h, record in records.items():
        origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
        np.testing.assert_array_equal(record["origins"], origins)
        actual, predicted = np.concatenate(record["actual"]), np.concatenate(record["predicted"])
        if not np.isfinite(predicted).all():
            raise ValueError("Forecasts cannot shrink beam coverage")
        mask = np.isfinite(actual)
        error = np.abs(predicted[mask]-actual[mask])
        zero = mask & (actual == 0)
        rows.append({"horizon": h, "mae": float(error.mean()),
                     "wape": float(error.sum()/np.abs(actual[mask]).sum()),
                     "n_valid": int(mask.sum()), "zero_fp_rate": float((predicted[zero] > 0).mean())})
        np.savez_compressed(directory/f"score_h{h}.npz", actual=actual, predicted=predicted,
                            origins=origins, beam_ids=np.asarray(panel.beam_ids))
    return rows


def run_study(config_path="configs/cell_attention.yaml", artifact_path="artifacts/cell_attention",
              report_prefix="cell_attention", initializer=None):
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required")
    cfg = load_config(config_path)
    root = Path(artifact_path)
    validate_output_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # The fused GPU inference path drifted versus CPU in the historical v1 study.
    # This affects future runs only; historical weights/forecasts remain immutable.
    configure_attention_math()
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    rows = []
    for fold in cfg["folds"]:
        for mode in cfg["modes"]:
            directory = root/fold["name"]/mode
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError("Inspect persisted checkpoints before resubmission")
            directory.mkdir(parents=True, exist_ok=True)
            set_neural_seed(cfg["seed"])
            preprocessing = GRUPreprocessor.fit(panel, fold["fit_end"], "B")
            preprocessing.save(directory/"preprocessor.npz")
            model = CellPatchAttention(cfg, preprocessing.mean[0], preprocessing.scale[0], mode)
            torch.save({"model": model.state_dict(), "seed": cfg["seed"],
                        "torch_initial_seed": torch.initial_seed(), "fold": fold, "mode": mode},
                       directory/"cold_initialization.pt")
            if initializer is not None:
                initializer(model, panel, preprocessing, fold, directory, cfg)
            metadata = {"protocol": cfg, "fold": fold, "mode": mode,
                "parameters": sum(p.numel() for p in model.parameters()), "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                "gpu_attention_fastpath": torch.backends.mha.get_fastpath_enabled(),
                "torch_initial_seed": torch.initial_seed(),
                "cold_initialization_sha256": sha256(directory/"cold_initialization.pt"),
                "holdout_used": False, "validation_504_671_used": False, "training_complete": False}
            (directory/"meta.json").write_text(json.dumps(metadata, indent=2))
            train = CellBatches(panel, preprocessing, train_origins_joint(fold["fit_end"]), "cuda")
            stopping = CellBatches(panel, preprocessing, evaluation_origins(fold["fit_end"], fold["stop_end"], 24), "cuda")
            print(json.dumps({"fold": fold["name"], "mode": mode, "parameters": metadata["parameters"],
                              "training_cell_windows": train.n_samples}), flush=True)
            result = train_gru(model, train, stopping, cfg, directory, cfg["seed"],
                run_identity={"fold": fold, "mode": mode, "architecture": cfg["protocol"]})
            model.load_state_dict(torch.load(directory/"best.pt", map_location="cuda", weights_only=True)["model"])
            metadata.update(training=result, training_complete=True,
                            loss_units="physical DLPRB; historical trainer field names contain scaled")
            (directory/"meta.json").write_text(json.dumps(metadata, indent=2))
            rows.extend({"fold": fold["name"], "mode": mode, **r} for r in score(model, panel, preprocessing, fold, directory))
            pd.DataFrame(rows).to_csv(f"reports/{report_prefix}_development.csv", index=False)
            print(pd.DataFrame(rows).tail(3).to_string(index=False), flush=True)
            del model, train, stopping
            torch.cuda.empty_cache()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))


def main():
    run_study()


if __name__ == "__main__":
    main()
