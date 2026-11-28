"""Hosted multi-corpus encoder pretraining followed by controlled cell fine-tuning."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_cell_attention import run_study
from src.data.io import sha256
from src.evaluation.origins import train_origins_joint
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.features.pretraining_batches import PretrainingBatches, SampledTargetBatches
from src.models.cell_pretraining import SourcePatchForecaster, transfer_encoder
from src.models.gru_train import train_gru
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training
from src.utils.seed import set_seed


class ContextNormalized:
    """Normalize both supervision and inputs using observed history only."""
    def __init__(self, base):
        self.base, self.device, self.n_samples = base, base.device, base.n_samples

    def batches(self, batch_size, shuffle_seed=None):
        for sequence, phase, target in self.base.batches(batch_size, shuffle_seed):
            observed = sequence[:, :, 1] == 0
            count = observed.sum(1, keepdim=True).clamp_min(1)
            mean = (sequence[:, :, 0]*observed).sum(1, keepdim=True)/count
            scale = (((sequence[:, :, 0]-mean).square()*observed).sum(1, keepdim=True)/count).sqrt().clamp_min(1e-5)
            normalized = sequence.clone()
            normalized[:, :, 0] = torch.where(observed, (sequence[:, :, 0]-mean)/scale, 0)
            yield normalized, phase, (target-mean)/scale


def main():
    require_cloud_training()
    cfg_path = "configs/cell_multisource.yaml"
    cfg = load_config(cfg_path)
    for case in cfg["cases"]:
        root = Path("artifacts/cell_multisource")/case

        def initialize(model, panel, preprocessing, fold, directory, config):
            saved_rng = torch.get_rng_state()
            saved_cuda = torch.cuda.get_rng_state_all()
            set_seed(config["seed"])
            source_model = SourcePatchForecaster(config)
            if case == "target_self_pretrain":
                prep = GRUPreprocessor.fit(panel, fold["fit_end"], "A")
                base = GRUBatches(panel, prep, "A", train_origins_joint(fold["fit_end"]),
                                  np.arange(len(panel.beam_ids)), "cuda")
                train = SampledTargetBatches(base, config["pretraining_samples_per_epoch"])
                diagnostic = SampledTargetBatches(base, 2048)
                pretrain_root = root/"pretraining"/fold["name"]
                provenance = {"original_fit_end": fold["fit_end"], "maximum_label_hour": fold["fit_end"],
                              "source": "original_fit_prefix_only"}
            else:
                reports, weights = config["source_reports"], config["source_weights"]
                overlap = json.loads(Path("reports/transfer_overlap_audit.json").read_text())
                if not overlap["pooled_training_permitted"] or sha256(Path("reports/multisource_preparation.json")) != overlap["source_preparation_sha256"]:
                    raise ValueError("Current radio overlap audit required")
                train = PretrainingBatches(reports, weights, config["pretraining_samples_per_epoch"], "cuda")
                diagnostic = PretrainingBatches(reports, weights, 2048, "cuda")
                pretrain_root = root/"pretraining"/"external"
                provenance = {"sources": train.provenance, "target_network_used": False}
            fixed = {**config, "batch_size": config["pretraining_batch_size"],
                     "max_epochs": config["pretraining_epochs"],
                     "early_stopping_patience": config["pretraining_epochs"]+1}
            identity = {"case": case, "provenance": provenance, "task": "past_context_normalized_source_forecasting"}
            meta_path = pretrain_root/"source_meta.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                if metadata["identity"] != identity or metadata["config"] != fixed:
                    raise ValueError("Existing pretraining identity differs")
            else:
                if (pretrain_root/"last.pt").exists():
                    raise RuntimeError("Inspect partial pretraining state before any restart")
                result = train_gru(source_model, ContextNormalized(train), ContextNormalized(diagnostic),
                    fixed, pretrain_root, config["seed"], run_identity=identity)
                if result["epochs_completed"] != config["pretraining_epochs"]:
                    raise ValueError("Fixed pretraining budget incomplete")
                meta_path.write_text(json.dumps({"identity": identity, "config": fixed,
                    "training": result, "checkpoint": "last.pt", "units": "dimensionless past-context scale",
                    "diagnostic_is_training_prefix_only": True}, indent=2))
            state_path = pretrain_root/"last.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            transferred = transfer_encoder(model, state["model"])
            torch.save({"model": model.state_dict(), "source_checkpoint_sha256": sha256(state_path),
                        "transferred_encoder_keys": transferred, "case": case, "fold": fold},
                       directory/"initialization.pt")
            torch.set_rng_state(saved_rng)
            torch.cuda.set_rng_state_all(saved_cuda)
            del source_model, train, diagnostic
            torch.cuda.empty_cache()

        run_study(cfg_path, str(root), f"cell_multisource_{case}", initialize)
    complete = Path("artifacts/cell_multisource")
    files = {p.relative_to(complete).as_posix(): sha256(p) for p in complete.rglob("*")
             if p.is_file() and p != complete/"manifest.json"}
    (complete/"manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
