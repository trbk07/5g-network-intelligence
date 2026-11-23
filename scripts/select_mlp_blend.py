"""Hosted-only MLP blend coefficient selection; no base-model retraining."""

import argparse
from pathlib import Path

from scripts.select_development_blend import run_selection
from src.data.io import sha256
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-root",type=Path,required=True)
    parser.add_argument("--hurdle-root",type=Path,required=True)
    args = parser.parse_args()
    cfg = load_config("configs/mlp_blend.yaml")
    for name,root in (("mlp",args.mlp_root),("hurdle",args.hurdle_root)):
        if sha256(root/"manifest.json") != cfg[f"{name}_manifest_sha256"]:
            raise ValueError("Pinned source manifest differs")
    run_selection(cfg,args.mlp_root,args.hurdle_root,"artifacts/mlp_blend","mlp_blend",component="mlp")


if __name__ == "__main__":
    main()
