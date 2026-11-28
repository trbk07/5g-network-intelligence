"""Hosted TimeXer-inspired paired ablation; no original holdout reuse."""

import json
import os
from pathlib import Path

import torch

from scripts.train_multisource_neural import validate_output_storage
from scripts.train_temporal_mlp import train_study
from src.data.io import sha256
from src.models.cell_attention import configure_attention_math
from src.models.past_covariate_attention import PastCovariateAttention
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    root = Path("artifacts/past_covariate_attention")
    validate_output_storage(root)
    if root.exists():
        raise FileExistsError("Inspect existing job/checkpoints before any repeat fitting")
    configure_attention_math()
    config_path = "configs/past_covariate_attention.yaml"
    cfg = load_config(config_path)
    if cfg["cases"] != ["no_covariates", "past_covariates"] or cfg["variants"] != ["B"]:
        raise ValueError("Declared paired four-KPI ablation required")
    for case in cfg["cases"]:
        case_root = root/case

        def factory(config, variant):
            if variant != "B":
                raise ValueError("Both ablations retain identical four-KPI preprocessing")
            return PastCovariateAttention(config, use_covariates=case == "past_covariates")

        def initialize(model, panel, preprocessing, fold, config):
            directory = case_root/fold["name"]/"B"
            model.configure_target_scale(float(preprocessing.mean[0]), float(preprocessing.scale[0]))
            cold = directory/"cold_initialization.pt"
            torch.save({"model": model.state_dict(), "seed": config["seed"],
                        "torch_initial_seed": torch.initial_seed(), "case": case, "fold": fold}, cold)
            (directory/"adaptation.json").write_text(json.dumps({"case": case,
                "paper_reproduction": False, "cold_initialization_sha256": sha256(cold),
                "gpu_attention_fastpath": torch.backends.mha.get_fastpath_enabled(),
                "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                "target_scale_fit_end": preprocessing.fit_end, "future_covariates_used": False,
                "units": "standardized training/stopping; nonnegative physical DLPRB scoring"}, indent=2))
            print(json.dumps({"case": case, "fold": fold["name"], "parameters": sum(p.numel() for p in model.parameters()),
                              "cold_initialization_sha256": sha256(cold)}), flush=True)
            return model

        train_study(config_path, str(case_root), f"past_covariate_attention_{case}",
                    architecture="TimeXer-inspired-independent-adaptation/"+case,
                    model_factory=factory, model_initializer=initialize)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
