"""Hosted paired joint/available-label training; existing stopping/scoring unchanged."""

import json
from pathlib import Path

import numpy as np

from scripts.train_cell_attention import run_study
from scripts.configure_partial_supervision import protocols
from scripts.train_multisource_neural import validate_output_storage
from src.data.io import sha256
from src.evaluation.origins import train_origins, train_origins_joint
from src.features.cell_inputs import CellBatches
from src.features.partial_cell_inputs import PartialCellBatches
from src.models.partial_supervision import horizon_balanced_l1
from src.utils.runtime import require_cloud_training
from src.utils.config import load_config


def supervision_counts(panel, origins, label_end):
    """Count only observed in-prefix labels; never inspect targets beyond cutoff."""
    prefix = panel.values[:label_end+1, :, 0]
    return {str(h): int(np.isfinite(prefix[origins[origins+h <= label_end]+h]).sum()) for h in (1, 6, 24)}


def balanced_objective(model, history, phase, target):
    """Shared objective for both arms; no loss/weight selection from scores."""
    return horizon_balanced_l1(model(history, phase), target)


def main():
    require_cloud_training()
    declared = protocols(load_config("configs/cell_attention_v2.yaml"))
    for arm, expected in declared.items():
        if load_config(f"configs/partial_supervision_{arm}.yaml") != expected:
            raise ValueError("Matched partial-supervision protocol differs")
    root = Path("artifacts/partial_supervision")
    validate_output_storage(root)
    if root.exists():
        raise FileExistsError("Inspect existing job before any repeat fitting")
    for arm in ("joint", "available"):
        def factory(panel, preprocessor, fold, directory):
            origins = (train_origins_joint(fold["fit_end"]) if arm == "joint"
                       else train_origins(fold["fit_end"], 1))
            batcher = (CellBatches(panel, preprocessor, origins, "cuda") if arm == "joint"
                       else PartialCellBatches(panel, preprocessor, origins, fold["fit_end"], "cuda"))
            counts = {"arm": arm, "fit_end": fold["fit_end"], "horizon_valid_labels": supervision_counts(
                panel, origins, fold["fit_end"]), "training_cell_windows": batcher.n_samples,
                "training_objective": "mean_of_observed_per_horizon_batch_MAEs",
                "stopping_policy": "unchanged_complete_joint_windows"}
            (directory/"supervision_counts.json").write_text(json.dumps(counts, indent=2))
            print(json.dumps(counts), flush=True)
            return batcher

        run_study(f"configs/partial_supervision_{arm}.yaml", str(root/arm), f"partial_supervision_{arm}",
                  training_batches=factory, training_loss=balanced_objective)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
