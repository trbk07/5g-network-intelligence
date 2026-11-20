"""Create a checksum-verified 24-beam real-data subset. Never train or publish raw data."""

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

from src.data.io import sha256
from src.data.panel import Panel, load_panel, save_panel
from src.inference.freeze import verify_artifacts


def install_archive(archive_path, destination="."):
    """Install the exact pinned demo package without raw data, training or model rebuilding."""
    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    package = json.loads(Path("reports/demo_asset_package.json").read_text(encoding="utf-8"))
    if sha256(archive_path) != package["sha256"]:
        raise ValueError("Demo package checksum mismatch")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if any(
            not m.name.startswith("demo_assets/")
            or not (destination / m.name).resolve().is_relative_to(destination)
            or m.issym()
            or m.islnk()
            for m in members
        ):
            raise ValueError("Unsafe demo package member")
        archive.extractall(destination, filter="data")
    from src.inference.predictor import Predictor

    predictor = Predictor(destination / "demo_assets")
    if (
        predictor.version != package["model_version"]
        or len(predictor.store.index) != package["beams_loaded"]
    ):
        raise ValueError("Installed demo identity mismatch")
    print("Installed verified frozen demo package; no original data or fitting required")


def prepare(
    panel_path="data/processed/panel.npz", champion="artifacts/champion", output="demo_assets"
):
    """Small actual panel plus original global models; no per-beam input statistics."""
    manifest = verify_artifacts(champion)
    meta = json.loads((Path(champion) / "champion_meta.json").read_text())
    frozen = json.loads(Path("reports/freeze_manifest.json").read_text())
    if (
        manifest["freeze_commit"] != frozen["commit"]
        or manifest["frozen_config_sha256"] != frozen["config_sha256"]
    ):
        raise ValueError("Assets differ from frozen champion")
    from src.utils.config import load_config

    config = load_config("configs/champion.yaml")
    if sha256(Path(panel_path)) != config["panel_sha256"]:
        raise ValueError("Canonical panel checksum mismatch")
    panel = load_panel(panel_path)
    beams = tuple(
        f"{bs}_{cell}_{beam}" for bs in (0, 5, 15, 29) for cell in (0, 1, 2) for beam in (0, 8)
    )
    indices = [panel.beam_ids.index(b) for b in beams]
    subset = Panel(
        panel.values[:, indices, :].copy(),
        panel.time.copy(),
        beams,
        {
            "dataset_revision": meta["dataset_revision"],
            "scope": "24 real beams from the internal historical replay panel",
            "is_replay": True,
        },
    )
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(champion, target / "champion", dirs_exist_ok=True)
    save_panel(subset, target / "replay.npz")
    benchmark = pd.read_csv("reports/champion_benchmark_holdout.csv")
    benchmark = benchmark[benchmark.load_tier == "All"][["model", "horizon", "mae", "rmse", "wape"]]
    files = {
        p.relative_to(target).as_posix(): sha256(p)
        for p in target.rglob("*")
        if p.is_file() and p.name != "manifest.json" and p.name != "README.md"
    }
    result = {
        "dataset_revision": meta["dataset_revision"],
        "dataset_scope": subset.metadata["scope"],
        "beam_ids": list(beams),
        "fit_end": meta["fit_end"],
        "frozen_config_sha256": manifest["frozen_config_sha256"],
        "freeze_commit": manifest["freeze_commit"],
        "license": "MIT; cite NETOP GOTS-F dataset authors",
        "files": files,
        "benchmark": benchmark.to_dict("records"),
    }
    (target / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Prepared", len(beams), "real replay beams; checksums recorded; no training")
    return result


def download():
    """Authenticated private Kaggle outputs are real existing sources, not an invented release URL."""
    local = Path(".kaggle-venv/Scripts/kaggle.exe")
    executable = str(local) if local.exists() else shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "Install Kaggle CLI and authenticate as a collaborator of the private execution notebooks"
        )
    cache = Path(".cache/demo-download")
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "kernels",
            "output",
            "truongbaokien/5g-network-intelligence-week6-final",
            "-p",
            str(cache),
            "--file-pattern",
            "week6_results.tar.gz",
            "-q",
        ],
        check=True,
    )
    archive = cache / "week6_results.tar.gz"
    evidence = json.loads(Path("reports/week6_evidence_manifest.json").read_text())
    if sha256(archive) != evidence["archive_sha256"]:
        raise ValueError("Downloaded final archive checksum differs from pinned evidence")
    with tarfile.open(archive) as handle:
        # Only frozen model files are extracted; reject links and traversal before extraction.
        members = [m for m in handle.getmembers() if m.name.startswith("artifacts/champion/")]
        if any(
            not (Path.cwd() / m.name).resolve().is_relative_to(Path.cwd()) or m.issym() or m.islnk()
            for m in members
        ):
            raise ValueError("Unsafe archive member")
        handle.extractall(Path.cwd(), members=members, filter="data")
    subprocess.run(
        [
            executable,
            "kernels",
            "output",
            "truongbaokien/5g-network-intelligence-week5",
            "-p",
            str(cache),
            "--file-pattern",
            "week5/data/processed/panel.npz",
            "-q",
        ],
        check=True,
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache / "week5/data/processed/panel.npz", "data/processed/panel.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="Download pinned private Kaggle outputs first"
    )
    parser.add_argument(
        "--demo-archive", help="Install the pinned packaged demo directly, no raw data required"
    )
    args = parser.parse_args()
    if args.demo_archive:
        if args.download:
            parser.error("--demo-archive and --download are mutually exclusive")
        install_archive(args.demo_archive)
    else:
        if args.download:
            download()
        prepare()
