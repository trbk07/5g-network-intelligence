"""Verify cloud-selected MLP coefficients and fixed confirmation without selection."""

from pathlib import Path

from scripts.summarize_development_blend import verify_blend
from src.data.io import sha256
from src.utils.config import load_config


def main():
    cfg=load_config("configs/mlp_blend.yaml")
    for name,path in (("mlp","artifacts/temporal_mlp"),("hurdle","artifacts/hurdle_research")):
        if sha256(Path(path)/"manifest.json") != cfg[f"{name}_manifest_sha256"]:
            raise ValueError("Pinned component manifest differs")
    verify_blend("configs/mlp_blend.yaml","artifacts/mlp_blend","artifacts/temporal_mlp","mlp_blend","mlp")


if __name__ == "__main__":
    main()
