"""Hosted multivariate direct-versus-hurdle study with MAE-only stopping."""

import torch

from scripts.train_multisource_neural import validate_output_storage
from scripts.train_temporal_mlp import train_study
from src.models.neural_hurdle import NeuralHurdleMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from pathlib import Path


def main():
    require_cloud_training()
    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required")
    root = Path("artifacts/neural_hurdle")
    validate_output_storage(root)
    cfg_path = "configs/neural_hurdle.yaml"
    cfg = load_config(cfg_path)
    for kind in cfg["cases"]:
        def factory(config, variant):
            if variant != "B":
                raise ValueError("This pilot requires all four original KPIs")
            return NeuralHurdleMLP(4, config["hidden_size"], config["dropout"], kind=kind)

        def initialize(model, panel, preprocessing, fold, config):
            if preprocessing.fit_end != fold["fit_end"]:
                raise ValueError("Training-prefix preprocessing required")
            with torch.no_grad():
                model.zero_scaled.fill_(-preprocessing.mean[0]/preprocessing.scale[0])
                model.target_mean.fill_(preprocessing.mean[0])
                model.target_scale.fill_(preprocessing.scale[0])
            return model

        train_study(cfg_path, str(root/kind), f"neural_hurdle_{kind}",
                    architecture=f"multivariate-neural-{kind}-v1", model_factory=factory,
                    model_initializer=initialize,
                    training_loss=lambda model, sequence, phase, target:
                        model.training_loss(sequence, phase, target))


if __name__ == "__main__":
    main()
