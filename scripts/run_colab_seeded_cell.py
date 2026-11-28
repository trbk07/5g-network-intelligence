"""Hosted Colab seed-29 baseline with public-source acquisition and Drive checkpoints."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import urllib.request
import zipfile


def verify_code(archive_path, expected):
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Code archive checksum differs")
    with zipfile.ZipFile(archive_path) as archive:
        names = []
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or ":" in member.filename or "\\" in member.filename
                    or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe code archive member")
            names.append(member.filename)
        manifest = json.loads(archive.read("source_manifest.json"))
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"source_manifest.json"}:
            raise ValueError("Code members differ from manifest")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Code member checksum differs")
    return manifest


def main():
    if os.name == "nt" or "COLAB_RELEASE_TAG" not in os.environ or not Path("/content").is_dir():
        raise RuntimeError("Hosted Colab only; local training forbidden")
    archive_path = Path("/content/seeded_cell_code.zip")
    manifest = verify_code(archive_path, os.environ["CELL_CODE_SHA256"])
    drive = Path("/content/drive/MyDrive")
    if not drive.is_dir():
        raise RuntimeError("Mounted Drive required")
    run = drive/"5g-network-intelligence/checkpoints"/("cell_seed29_"+manifest["source_commit"][:12])
    run.mkdir(parents=True, exist_ok=True)
    state_path = run/"runtime_state.json"
    if state_path.exists() and json.loads(state_path.read_text())["status"] not in ("preparing", "prepared"):
        raise RuntimeError("Existing training state; inspect before any repeat fitting")
    state = {"status": "preparing", "source_commit": manifest["source_commit"],
             "public_data_downloaded_by_runtime": True, "local_beam_payload_uploaded": False,
             "training_started": False, "seed": 29}
    state_path.write_text(json.dumps(state, indent=2))
    project = Path("/content/cell_seed29")
    project.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(project)  # Every member/path/hash already verified.
    (project/"artifacts").mkdir(exist_ok=True)
    for link, target in ((project/"artifacts/cell_attention_seed29", run/"artifacts/cell_attention_seed29"),
                         (project/"reports", run/"reports")):
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError("Unexpected storage destination")
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ["SOURCE_GIT_COMMIT"] = manifest["source_commit"]
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(20):
            print("[Colab] Seed-29 worker active; outputs on Drive.", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy", "pandas", "pyyaml",
                        "lightgbm>=4.6,<5", "scikit-learn"], check=True)
        import numpy as np
        import torch
        from src.data.io import sha256
        from src.data.panel import build_panel, save_panel
        if not torch.cuda.is_available():
            raise RuntimeError("Hosted GPU required")
        public = json.loads(Path("public_sources.json").read_text())
        paths = {}
        for kpi, item in public["train_files"].items():
            target = project/"data/raw"/item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != item["sha256"]:
                print(json.dumps({"public_source_file": item["path"], "status": "downloading"}), flush=True)
                partial = target.with_suffix(".part")
                with urllib.request.urlopen(item["url"], timeout=120) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output, length=8*1024**2)
                if sha256(partial) != item["sha256"]:
                    raise ValueError("Public publisher checksum differs")
                partial.replace(target)
            paths[kpi] = target
        panel = build_panel(paths, {"dataset_revision": public["revision"], "license": "mit", "segment": "train"})
        actual_prefix = hashlib.sha256(panel.values[:504].tobytes()).hexdigest()
        if list(panel.values.shape) != [840, 2880, 4] or actual_prefix != public["prefix_values_sha256"]:
            raise ValueError("Public rebuild differs from frozen canonical prefix; do not train")
        panel.values[504:] = np.nan
        processed = Path("data/processed")
        save_panel(panel, processed/"panel.npz")
        (processed/"split.json").write_text(json.dumps(public["frozen_split"], indent=2))
        (run/"reports/public_source_rebuild.json").write_text(json.dumps({
            "repo_id": public["repo_id"], "revision": public["revision"], "license": "MIT",
            "prefix_values_sha256": actual_prefix, "values_504_onward_redacted": True,
            "canonical_prefix_matches": True, "local_beam_payload_uploaded": False}, indent=2))
        state.update(status="prepared", torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0))
        state_path.write_text(json.dumps(state, indent=2))
        if os.environ.get("CELL_PREPARE_ONLY", "true") == "true":
            print(json.dumps(state), flush=True)
            return
        with (run/"reports/cell_attention_seed29_training.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "scripts.train_cell_attention_seed29"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            state.update(status="running", training_started=True, training_pid=process.pid)
            state_path.write_text(json.dumps(state, indent=2))
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        state.update(status="complete" if status == 0 else "failed", exit_code=status)
        state_path.write_text(json.dumps(state, indent=2))
        with tarfile.open(run/"cell_attention_seed29_results.tar.gz", "w:gz") as archive:
            for folder in ("artifacts/cell_attention_seed29", "reports"):
                archive.add(project/folder, arcname=folder)
        if status:
            raise RuntimeError("Inspect persistent training log before any repeat fitting")
    finally:
        finished.set()


if __name__ == "__main__":
    main()
