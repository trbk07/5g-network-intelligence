"""Hosted-only native multivariate LoRA; checkpoint selection is a separate stage."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from scripts.train_multisource_neural import validate_output_storage
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.features.chronos2_inputs import training_prefix, native_tasks
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config("configs/chronos2_lora.yaml")
    model_cfg = load_config(cfg["model_config"])
    for package, key in (("chronos-forecasting", "chronos_version"),
                         ("transformers", "transformers_version"), ("peft", "peft_version")):
        if importlib.metadata.version(package) != str(cfg[key]):
            raise RuntimeError("Pinned fine-tuning package version differs: "+package)
    # Explicit import/version check prevents the library's silent full-finetuning fallback.
    import peft  # noqa: F401
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required for foundation-model fine-tuning")
    if sha256(args.model_root/model_cfg["weights_file"]) != model_cfg["weights_sha256"]:
        raise ValueError("Pinned synthetic-pretrained safetensors checksum differs")
    if sha256(args.model_root/"config.json") != model_cfg["config_sha256"]:
        raise ValueError("Pinned model configuration checksum differs")
    configuration = json.loads((args.model_root/"config.json").read_text())
    if "auto_map" in configuration:
        raise ValueError("Remote custom code is not permitted")
    root = Path("artifacts/chronos2_lora")
    validate_output_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    panel, _ = validation_inputs(load_config("configs/data.yaml"))

    class CheckLoRA(TrainerCallback):
        def on_train_begin(self, arguments, state, control, model=None, **kwargs):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            if not getattr(model, "peft_config", None) or not 0 < trainable/total <= cfg["maximum_trainable_fraction"]:
                raise RuntimeError("LoRA missing or unexpectedly large trainable parameter fraction")
            print(json.dumps({"trainable_parameters": trainable, "total_parameters": total,
                              "mode_verified": "lora"}), flush=True)

    for fold in cfg["folds"]:
        output = root/fold["name"]
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Existing checkpoint state; inspect before any resubmission")
        output.mkdir(parents=True, exist_ok=True)
        inputs = training_prefix(panel.values, fold["fit_end"])
        set_seed(cfg["seed"])
        pipeline = Chronos2Pipeline.from_pretrained(str(args.model_root), device_map="cuda",
                                                    torch_dtype=torch.float32, local_files_only=True)
        metadata = {"protocol": cfg, "fold": fold, "model_revision": model_cfg["revision"],
                    "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                    "weights_sha256": model_cfg["weights_sha256"], "input_shape": list(inputs.shape),
                    "maximum_input_hour": fold["fit_end"], "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(0), "scoring_performed": False,
                    "checkpoint_selection_pending": True, "holdout_used": False,
                    "validation_504_671_used": False}
        (output/"training_protocol.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"fold": fold["name"], "training_prefix_shape": list(inputs.shape),
                          "maximum_training_label_hour": fold["fit_end"]}), flush=True)
        tuned = pipeline.fit(inputs=native_tasks(inputs), prediction_length=cfg["prediction_length"],
            validation_inputs=None, finetune_mode="lora", lora_config=cfg["lora"],
            context_length=cfg["context_length"], min_past=cfg["min_past"],
            learning_rate=cfg["learning_rate"], num_steps=cfg["max_steps"],
            batch_size=cfg["batch_size_variates"], output_dir=output/"trainer",
            callbacks=[CheckLoRA()], seed=cfg["seed"], data_seed=cfg["seed"],
            save_strategy="steps", save_steps=cfg["checkpoint_steps"], save_total_limit=None,
            save_only_model=False, save_safetensors=True, logging_steps=25,
            load_best_model_at_end=False, eval_strategy="no")
        for step in cfg["selection_checkpoints"]:
            checkpoint = output/"trainer"/f"checkpoint-{step}"
            if not (checkpoint/"adapter_model.safetensors").exists() or not (checkpoint/"optimizer.pt").exists():
                raise RuntimeError("Required LoRA/optimizer checkpoint missing")
        metadata["fixed_step_training_complete"] = True
        (output/"training_protocol.json").write_text(json.dumps(metadata, indent=2))
        del tuned, pipeline, inputs
        torch.cuda.empty_cache()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name != "manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files, "checkpoint_selection_pending": True}, indent=2))


if __name__ == "__main__":
    main()
