"""Hosted fixed-budget external pretraining and matched target-only ablations."""

import json
import os
from pathlib import Path

import torch

from scripts.train_temporal_mlp import train_study
from src.data.io import sha256
from src.features.pretraining_batches import PretrainingBatches
from src.models.gru_train import train_gru
from src.models.temporal_mlp import TemporalResidualMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed


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
    if "COLAB_RELEASE_TAG" in os.environ:
        drive = Path("/content/drive/MyDrive").resolve()
        if not drive.is_dir() or not root.resolve().is_relative_to(drive):
            raise RuntimeError("Colab checkpoints must resolve to mounted Drive before fitting")
        if not Path("reports").resolve().is_relative_to(drive):
            raise RuntimeError("Colab reports must resolve to mounted Drive before fitting")
    root.mkdir(parents=True, exist_ok=True)
    for case in cfg["cases"]:
        set_seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.use_deterministic_algorithms(True)
        pretrained = None
        if case != "target_only":
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

        # All cases use identical fold boundaries, target fit budget and stopping protocol.
        train_study(cfg_path, str(root/case), f"multisource_neural_{case}",
                    architecture="multisource-univariate-mlp-v1", model_factory=factory)
    files = {p.relative_to(root).as_posix(): sha256(p)
             for p in root.rglob("*") if p.is_file() and p != root/"manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files, "protocol": cfg}, indent=2))


if __name__ == "__main__":
    main()
