"""Compatibility entry point: build a readable cloud notebook with real Python cells."""
import shutil
from pathlib import Path

from scripts.build_gru_review_notebook import build as build_review


def build() -> None:
    """Write the cloud entry notebook without embedding encoded source."""
    source = build_review()
    target = Path(__file__).resolve().parents[1] / "notebooks/00_cloud_standalone.ipynb"
    shutil.copyfile(source, target)
    print(f"Readable cloud notebook ready: {target}")


if __name__ == "__main__":
    build()
