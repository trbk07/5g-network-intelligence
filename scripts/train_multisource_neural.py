"""Hosted fixed-budget external pretraining and matched target-only ablations."""

import json
import os
from pathlib import Path

import torch
import numpy as np

from scripts.train_temporal_mlp import train_study
from src.data.io import sha256
from src.features.pretraining_batches import PretrainingBatches, SampledTargetBatches
from src.features.gru_inputs import GRUBatches
from src.evaluation.origins import train_origins_joint
from src.models.gru_train import train_gru
from src.models.temporal_mlp import TemporalResidualMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed

KAGGLE_WORKING = Path("/kaggle/working")
DRIVE_ROOT = Path("/content/drive/MyDrive")


def validate_output_storage(root, reports=Path("reports")):
    """Prefer the actual Kaggle filesystem over Colab markers inherited by images."""
    if KAGGLE_WORKING.is_dir():
        output = KAGGLE_WORKING.resolve()
        if not all(Path(path).resolve().is_relative_to(output) for path in (root, reports)):
            raise RuntimeError("Kaggle checkpoints and reports must stay in Output")
    elif "COLAB_RELEASE_TAG" in os.environ:
        drive = DRIVE_ROOT.resolve()
        if not drive.is_dir() or not all(Path(path).resolve().is_relative_to(drive)
                                        for path in (root, reports)):
            raise RuntimeError("Colab checkpoints and reports must resolve to mounted Drive before fitting")


def source_cases():
    radio = [f"reports/{name}_preparation.json" for name in
             ("pm01_nr1800", "pm03_nr2100", "pm03_nr3500")]
    cesnet = ["reports/cesnet_preparation.json"]
    return {"radio": (radio, [1, 1, 1]), "cesnet": (cesnet, [1]),
            "radio_and_cesnet": (radio+cesnet, [1, 1, 1, 3])}


def main():
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required for all pretraining and fine-tuning")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    cfg_path = "configs/multisource_neural_pilot.yaml"
    cfg = load_config(cfg_path)
    overlap_path = Path("reports/transfer_overlap_audit.json")
    overlap = json.loads(overlap_path.read_text())
    if (not overlap["pooled_training_permitted"] or
            sha256(Path("reports/multisource_preparation.json")) != overlap["source_preparation_sha256"]):
        raise ValueError("Current radio overlap audit required")
    root = Path("artifacts/multisource_neural")
    validate_output_storage(root)
    root.mkdir(parents=True, exist_ok=True)
    for case in cfg["cases"]:
        set_seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.use_deterministic_algorithms(True)
        pretrained = None
        if case in source_cases():
            reports, weights = source_cases()[case]
            train = PretrainingBatches(reports, weights, cfg["external_samples_per_epoch"], "cuda")
            diagnostic = PretrainingBatches(reports, weights, 2048, "cuda")
            directory = root / "external" / case
            model = TemporalResidualMLP(1, cfg["hidden_size"], cfg["dropout"])
            # Fixed budget: the in-prefix diagnostic cannot select an epoch.
            external_cfg = {**cfg, "max_epochs": cfg["external_epochs"],
                            "early_stopping_patience": cfg["external_epochs"]+1}
            summary = train_gru(model, train, diagnostic, external_cfg, directory, cfg["seed"],
                                resume=(directory/"last.pt").exists(),
                                run_identity={"case": case, "sources": train.provenance})
            if summary["epochs_completed"] != cfg["external_epochs"]:
                raise ValueError("Fixed external budget incomplete")
            state = torch.load(directory/"last.pt", map_location="cpu", weights_only=True)
            pretrained = state["model"]
            (directory/"meta.json").write_text(json.dumps({"case": case, "training": summary,
                "sources": train.provenance, "checkpoint_selected": "last.pt",
                "protocol": cfg, "torch_version": str(torch.__version__),
                "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                "gpu": torch.cuda.get_device_name(0),
                "diagnostic_is_training_prefix_not_generalization": True,
                "target_network_used_during_pretraining": False}, indent=2))
            del model, train, diagnostic
            torch.cuda.empty_cache()

        def factory(config, variant):
            if variant != "A":
                raise ValueError("Pilot transfers a univariate representation only")
            model = TemporalResidualMLP(1, config["hidden_size"], config["dropout"])
            if pretrained is not None:
                model.load_state_dict(pretrained, strict=True)
            return model

        def initialize(model, panel, preprocessing, fold, config):
            def record_initialization(source_checkpoint=None):
                path = root/case/fold["name"]/"A"/"initialization.pt"
                if path.exists():
                    raise FileExistsError("Refusing to overwrite initialization evidence")
                torch.save({"model": model.state_dict(), "case": case, "fold": fold,
                            "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
                            "source_checkpoint_sha256": sha256(source_checkpoint)
                            if source_checkpoint else None}, path)

            if case != "target_self_pretrain":
                record_initialization(root/"external"/case/"last.pt" if case in source_cases() else None)
                return model
            if preprocessing.fit_end != fold["fit_end"]:
                raise ValueError("Self-pretraining preprocessing must match fold fit cutoff")
            origins = train_origins_joint(fold["fit_end"], stride=config["training_stride"])
            if int(origins.max())+24 > fold["fit_end"]:
                raise ValueError("Self-pretraining target crosses fold fit boundary")
            base = GRUBatches(panel, preprocessing, "A", origins,
                              np.arange(len(panel.beam_ids)), "cuda")
            sampled = SampledTargetBatches(base, config["external_samples_per_epoch"])
            diagnostic = SampledTargetBatches(base, 2048)
            directory = root/"self_pretraining"/fold["name"]
            fixed = {**config, "max_epochs": config["external_epochs"],
                     "early_stopping_patience": config["external_epochs"]+1}
            fine_tune_rng = torch.get_rng_state()
            fine_tune_cuda_rng = torch.cuda.get_rng_state_all()
            summary = train_gru(model, sampled, diagnostic, fixed, directory, config["seed"],
                                resume=(directory/"last.pt").exists(),
                                run_identity={"case": case, "fold": fold,
                                              "fit_end": preprocessing.fit_end})
            if summary["epochs_completed"] != config["external_epochs"]:
                raise ValueError("Fixed target self-pretraining budget incomplete")
            state = torch.load(directory/"last.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            (directory/"meta.json").write_text(json.dumps({"case": case, "fold": fold,
                "training": summary, "fit_end": preprocessing.fit_end,
                "maximum_supervised_hour": int(origins.max())+24,
                "checkpoint_selected": "last.pt", "protocol": config,
                "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                "diagnostic_is_training_prefix_not_generalization": True}, indent=2))
            # Fine-tuning starts a fresh optimizer and the same RNG stream as other cases.
            torch.set_rng_state(fine_tune_rng)
            torch.cuda.set_rng_state_all(fine_tune_cuda_rng)
            record_initialization(directory/"last.pt")
            return model

        # All cases use identical fold boundaries, target fit budget and stopping protocol.
        train_study(cfg_path, str(root/case), f"multisource_neural_{case}",
                    architecture="multisource-univariate-mlp-v1", model_factory=factory,
                    model_initializer=initialize)
    files = {p.relative_to(root).as_posix(): sha256(p)
             for p in root.rglob("*") if p.is_file() and p != root/"manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files, "protocol": cfg}, indent=2))


if __name__ == "__main__":
    main()
