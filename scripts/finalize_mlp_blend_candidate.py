"""Hosted fixed-epoch MLP refit with previously selected blend weights."""

import argparse
import json
from pathlib import Path

from scripts.finalize_blend_candidate import fixed_epochs, run_audit
from src.data.io import sha256
from src.models.temporal_mlp import TemporalResidualMLP
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-root",type=Path,required=True)
    parser.add_argument("--blend-root",type=Path,required=True)
    parser.add_argument("--hurdle-replay",type=Path,required=True)
    args=parser.parse_args()
    cfg=load_config("configs/mlp_blend_candidate_audit.yaml")
    base=load_config("configs/temporal_mlp.yaml")
    if sha256(args.mlp_root/"manifest.json") != cfg["mlp_manifest_sha256"]:
        raise ValueError("Pinned MLP development manifest differs")
    selection=json.loads((args.blend_root/"selection.json").read_text())
    variants=sorted({choice["variant"] for choice in selection["choices"].values()})
    if variants != cfg["expected_selected_variants"]:
        raise ValueError("Selected variants differ from frozen audit")
    for variant in variants:
        metadata=[json.loads((args.mlp_root/fold["name"]/variant/"meta.json").read_text()) for fold in base["folds"]]
        if fixed_epochs(metadata) != cfg["expected_fixed_epochs"][variant]:
            raise ValueError("Development epoch counts differ from frozen audit")

    def factory(variant):
        return TemporalResidualMLP(1 if variant=="A" else 4,base["hidden_size"],base["dropout"])

    run_audit(cfg,base,args.mlp_root,args.blend_root,args.hurdle_replay,
              "artifacts/mlp_blend_candidate","mlp_blend_candidate",component="mlp",model_factory=factory)


if __name__ == "__main__":
    main()
