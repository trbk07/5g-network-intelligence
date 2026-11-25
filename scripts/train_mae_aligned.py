"""Run four hosted MAE-aligned fits, reusing previous controls without refitting."""

import json
from pathlib import Path

import torch

from scripts.train_multisource_neural import validate_output_storage
from scripts.train_temporal_mlp import train_study
from src.data.io import sha256
from src.models.mae_aligned import MAEAlignedMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required")
    root = Path("artifacts/mae_aligned")
    validate_output_storage(root)
    cfg_path = "configs/mae_aligned.yaml"
    cfg = load_config(cfg_path)
    for mode in cfg["cases"]:
        def factory(config, variant):
            if variant != "B":
                raise ValueError("Four original KPI channels required")
            return MAEAlignedMLP(4, config["hidden_size"], config["dropout"],
                                 mode=mode, auxiliary_weight=config["auxiliary_weight"])

        def initialize(model, panel, preprocessing, fold, config):
            if preprocessing.fit_end != fold["fit_end"]:
                raise ValueError("Fit-prefix preprocessing required")
            with torch.no_grad():
                model.zero_scaled.fill_(-preprocessing.mean[0]/preprocessing.scale[0])
                model.target_mean.fill_(preprocessing.mean[0])
                model.target_scale.fill_(preprocessing.scale[0])
            return model

        train_study(cfg_path, str(root/mode), f"mae_aligned_{mode}",
                    architecture=f"mae-aligned-{mode}-v1", model_factory=factory,
                    model_initializer=initialize,
                    training_loss=lambda model, sequence, phase, target:
                        model.training_loss(sequence, phase, target))
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
             if p.is_file() and p != root/"manifest.json"}
    (root/"manifest.json").write_text(json.dumps({"files": files, "protocol": cfg}, indent=2))


if __name__ == "__main__":
    main()
