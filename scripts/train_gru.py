"""GPU-only GRU-A/B training with a shared, training-stratified beam sample."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.evaluation.origins import evaluation_origins, train_origins_joint
from src.evaluation.validation import validation_inputs
from src.features.beam_sampling import sample_beams
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.features.static_stats import compute_static
from src.models.gru import GRUNet
from src.models.gru_train import train_gru
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata
from src.utils.seed import set_seed
from src.utils.runtime import require_cloud_training


def main() -> None:
    """Train one requested variant/seed; use --pilot for exactly one measured epoch."""
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seconds-limit", type=float, default=None)
    parser.add_argument("--config", default="configs/gru.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GRU training requires a CUDA GPU; CPU training is disabled")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data_config, config = load_config(args.data_config), load_config(args.config)
    if args.seed not in config["seeds"] or args.variant not in config["variants"]:
        raise ValueError("Requested run is outside the frozen experiment list")
    panel, split = validation_inputs(data_config)
    preprocessor = GRUPreprocessor.fit(panel, split["fit"][1])
    selected = sample_beams(compute_static(panel, split["fit"][1]), config["sample_beams"],
                             config["sample_seed"])
    reports = Path(data_config["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    sample_manifest = {"fit_end": split["fit"][1], "seed": config["sample_seed"],
                       "indices": selected.tolist(), "beam_ids": [panel.beam_ids[i] for i in selected],
                       "method": "proportional joint tertiles of fit mean/std/active ratio"}
    (reports / "gru_sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2))
    train_origins = train_origins_joint(split["fit"][1], stride=config["training_stride"])
    val_origins = evaluation_origins(split["fit"][1], split["validation"][1], 24)
    train = GRUBatches(panel, preprocessor, args.variant, train_origins, selected, "cuda")
    validation = GRUBatches(panel, preprocessor, args.variant, val_origins,
                            np.arange(len(panel.beam_ids)), "cuda")
    directory = Path(config["artifact_dir"]) / args.variant / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    preprocessor.save(directory / "preprocessor.npz")
    if (directory / "last.pt").exists() and not args.resume:
        raise ValueError("Existing run requires --resume; refusing to overwrite checkpoint history")
    model = GRUNet(2 if args.variant == "A" else 8, config["hidden_size"],
                    config["num_layers"], config["dropout"])
    training = dict(config)
    if args.pilot:
        training["max_epochs"] = 1
    identity = {"variant": args.variant, "dataset_revision": data_config["revision"],
                "fit_end": split["fit"][1], "validation_end": split["validation"][1],
                "beam_sample": selected.tolist()}
    torch.cuda.reset_peak_memory_stats()
    summary = train_gru(model, train, validation, training, directory, args.seed, args.resume,
                        run_identity=identity, seconds_limit=args.seconds_limit)
    meta = {**collect_run_metadata({**data_config, "gru": config}), **summary,
            "variant": args.variant, "seed": args.seed, "architecture": {
                k: config[k] for k in ("hidden_size", "num_layers", "dropout")},
            "fit_end": split["fit"][1], "train_samples": train.n_samples,
            "validation_joint_samples": validation.n_samples, "pilot": args.pilot,
            "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
