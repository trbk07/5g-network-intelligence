"""Hosted stopping-only checkpoint selection and full original development replay."""

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from scripts.train_multisource_neural import validate_output_storage
from src.data.io import sha256
from src.evaluation.chronos2_replay import cached_origin
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def stopping_mae(pipeline, values, fold, cache_root, identity, batch_size):
    """One fixed joint-origin stopping mask across all three horizons and beams."""
    total, count = 0., 0
    for origin in evaluation_origins(fold["fit_end"], fold["stop_end"], 24):
        predicted = cached_origin(pipeline, values, int(origin), cache_root, identity, batch_size)
        actual = values[origin+np.array([1, 6, 24]), :, 0].T
        mask = np.isfinite(actual)
        total += float(np.abs(predicted[:, [0, 5, 23]][mask]-actual[mask]).sum())
        count += int(mask.sum())
    if not count:
        raise ValueError("No observed stopping targets")
    return {"mae": total/count, "n_valid": count}


def score_case(pipeline, panel, fold, stage, directory, identity, batch_size):
    """Forecast from causal history; score only labels within the fixed scoring end."""
    horizons = (1, 6, 24)
    origins, predictions, actuals = ({h: [] for h in horizons} for _ in range(3))
    for origin in range(fold["stop_end"], fold["score_end"]):
        predicted = cached_origin(pipeline, panel.values, origin, directory/"cache", identity, batch_size)
        for h in horizons:
            if origin+h <= fold["score_end"]:
                origins[h].append(origin)
                predictions[h].append(predicted[:, h-1])
                actuals[h].append(panel.values[origin+h, :, 0])
    rows = []
    for h in horizons:
        expected = evaluation_origins(fold["stop_end"], fold["score_end"], h)
        np.testing.assert_array_equal(origins[h], expected)
        predicted, actual = np.concatenate(predictions[h]), np.concatenate(actuals[h])
        mask = np.isfinite(actual)
        error = np.abs(predicted[mask]-actual[mask])
        denominator = float(np.abs(actual[mask]).sum())
        if not mask.any() or denominator <= 0 or not np.isfinite(predicted[mask]).all():
            raise ValueError("Invalid full-population scoring coverage/denominator")
        zero = mask & (actual == 0)
        rows.append({"fold": fold["name"], "stage": stage, "horizon": h,
                     "mae": float(error.mean()), "wape": float(error.sum()/denominator),
                     "n_valid": int(mask.sum()), "zero_fp_rate": float((predicted[zero] > 0).mean())})
        np.savez_compressed(directory/f"score_h{h}.npz", predicted=predicted, actual=actual,
                            origins=expected, beam_ids=np.asarray(panel.beam_ids))
    return rows


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--trained-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config("configs/chronos2_lora.yaml")
    model_cfg = load_config(cfg["model_config"])
    for package, key in (("chronos-forecasting", "chronos_version"),
                         ("transformers", "transformers_version"), ("peft", "peft_version")):
        if importlib.metadata.version(package) != str(cfg[key]):
            raise ValueError("Evaluation package differs from pinned protocol")
    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required for full foundation-model evaluation")
    for name, expected in ((model_cfg["weights_file"], model_cfg["weights_sha256"]),
                            ("config.json", model_cfg["config_sha256"])):
        if sha256(args.model_root/name) != expected:
            raise ValueError("Base weights/config differ")
    training_manifest = json.loads((args.trained_root/"manifest.json").read_text())
    for name, digest in training_manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
            raise ValueError("Unsafe trained artifact path")
        if sha256(args.trained_root/name) != digest:
            raise ValueError("Trained artifact checksum differs")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    root = Path("artifacts/chronos2_lora_evaluation")
    validate_output_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    identity = {"protocol": cfg, "training_manifest_sha256": sha256(args.trained_root/"manifest.json"),
                "base_sha256": model_cfg["weights_sha256"], "model_revision": model_cfg["revision"],
                "panel_prefix_sha256": hashlib.sha256(panel.values.tobytes()).hexdigest(),
                "beam_ids_sha256": hashlib.sha256(json.dumps(panel.beam_ids).encode()).hexdigest(),
                "torch": str(torch.__version__)}
    identity_path = root/"evaluation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing evaluation identity differs; refuse stale caches")
    identity_path.write_text(json.dumps(identity, indent=2))

    def load_pipeline(adapter=None):
        base = Chronos2Pipeline.from_pretrained(str(args.model_root), device_map="cuda",
                                                torch_dtype=torch.float32, local_files_only=True)
        if adapter is not None:
            config = json.loads((adapter/"adapter_config.json").read_text())
            if (config["peft_type"] != "LORA" or config["r"] != cfg["lora"]["r"] or config["lora_alpha"] != cfg["lora"]["lora_alpha"]
                    or set(config["target_modules"]) != set(cfg["lora"]["target_modules"])):
                raise ValueError("LoRA adapter configuration differs")
            model = PeftModel.from_pretrained(base.model, str(adapter), is_trainable=False, local_files_only=True)
            base = Chronos2Pipeline(model=model)
        base.model.eval()
        return base

    rows, stopping_rows = [], []
    for fold in cfg["folds"]:
        trained = args.trained_root/fold["name"]
        metadata = json.loads((trained/"training_protocol.json").read_text())
        if (metadata["protocol"] != cfg or metadata["fold"] != fold
                or not metadata["fixed_step_training_complete"] or metadata["holdout_used"]
                or metadata["validation_504_671_used"] or metadata["torch"] != str(torch.__version__)):
            raise ValueError("Training/evaluation temporal or runtime identity differs")
        if (metadata["maximum_input_hour"] != fold["fit_end"]
                or metadata["input_shape"] != [len(panel.beam_ids), 4, fold["fit_end"]+1]
                or metadata["weights_sha256"] != model_cfg["weights_sha256"]
                or metadata["model_revision"] != model_cfg["revision"]):
            raise ValueError("Native fine-tuning input boundary/base identity differs")
        for step in cfg["selection_checkpoints"]:
            checkpoint = trained/"trainer"/f"checkpoint-{step}"
            state = json.loads((checkpoint/"trainer_state.json").read_text())
            if state["global_step"] != step or state["max_steps"] != cfg["max_steps"]:
                raise ValueError("Persisted optimizer-step identity differs")
            if not all((checkpoint/name).is_file() for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth")):
                raise ValueError("Durable optimizer/scheduler/RNG state incomplete")
        directory = root/fold["name"]
        directory.mkdir(exist_ok=True)
        selection_path = directory/"selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["protocol"] != cfg or selection["fold"] != fold:
                raise ValueError("Existing stored checkpoint choice differs")
            stopping_rows.extend(selection["stopping_rows"])
        else:
            candidates = []
            for step in cfg["selection_checkpoints"]:
                adapter = trained/"trainer"/f"checkpoint-{step}"
                model_identity = {**identity, "adapter_sha256": sha256(adapter/"adapter_model.safetensors")}
                pipeline = load_pipeline(adapter)
                result = stopping_mae(pipeline, panel.values, fold, directory/f"stopping_step{step}",
                                      model_identity, cfg["batch_size_variates"])
                candidates.append({"fold": fold["name"], "step": step, **result})
                print(json.dumps(candidates[-1]), flush=True)
                del pipeline
                torch.cuda.empty_cache()
            chosen = min(candidates, key=lambda row: (row["mae"], row["step"]))
            selection = {"protocol": cfg, "fold": fold, "selected_step": chosen["step"],
                         "selection_data": "stopping_only", "stopping_rows": candidates}
            # Persist the choice before any scoring-segment target is used for metrics.
            selection_path.write_text(json.dumps(selection, indent=2))
            stopping_rows.extend(candidates)
    # Freeze both folds' stopping choices before reporting any scoring-segment metrics.
    pd.DataFrame(stopping_rows).to_csv("reports/chronos2_lora_stopping.csv", index=False)
    for fold in cfg["folds"]:
        trained = args.trained_root/fold["name"]
        directory = root/fold["name"]
        selection = json.loads((directory/"selection.json").read_text())
        for stage in ("zero_shot", "selected_lora"):
            adapter = (trained/"trainer"/f"checkpoint-{selection['selected_step']}" if stage == "selected_lora" else None)
            model_identity = {**identity, "adapter_sha256": sha256(adapter/"adapter_model.safetensors") if adapter else None}
            pipeline = load_pipeline(adapter)
            output = directory/stage
            output.mkdir(exist_ok=True)
            rows.extend(score_case(pipeline, panel, fold, stage, output, model_identity, cfg["batch_size_variates"]))
            del pipeline
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv("reports/chronos2_lora_development.csv", index=False)
        pd.DataFrame(stopping_rows).to_csv("reports/chronos2_lora_stopping.csv", index=False)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name != "manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files, "source_commit": os.environ.get("SOURCE_GIT_COMMIT")}, indent=2))


if __name__ == "__main__":
    main()
