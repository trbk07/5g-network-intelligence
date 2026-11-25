"""Launch the verified multisource pilot on Colab with durable Drive outputs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import zipfile


def main():
    if os.name == "nt" or "COLAB_RELEASE_TAG" not in os.environ or not Path("/content").is_dir():
        raise RuntimeError("Hosted Colab required; local training is forbidden")
    import torch
    from run_colab_tcn import verify_transport  # Uploaded shared checksum/path validator.

    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA required")
    drive = Path("/content/drive/MyDrive")
    if not drive.is_dir():
        raise RuntimeError("Mount Drive before launching")
    bundle = Path("/content/multisource_neural_source.zip")
    manifest = verify_transport(bundle, os.environ["MULTISOURCE_BUNDLE_SHA256"])
    if manifest["original_values_available_through"] != 503:
        raise ValueError("Development-only redacted panel required")
    run = drive/"5g-network-intelligence/checkpoints"/(
        "multisource_neural_"+manifest["source_commit"][:12])
    state_path = run/"runtime_state.json"
    if state_path.exists():
        raise RuntimeError("Existing execution state: inspect before resubmitting")
    project = Path("/content/multisource_neural")
    project.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(project)
    (project/"artifacts").mkdir(exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    for link, target in ((project/"artifacts/multisource_neural", run/"artifacts/multisource_neural"),
                         (project/"reports", run/"reports")):
        target.mkdir(parents=True, exist_ok=True)
        if link.name == "reports" and link.is_dir() and not link.is_symlink():
            # Preserve packaged provenance when replacing the local report folder with Drive.
            import shutil
            shutil.copytree(link, target, dirs_exist_ok=True)
            for child in link.iterdir():
                if not child.is_file():
                    raise ValueError("Unexpected nested packaged reports")
                child.unlink()
            link.rmdir()
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError("Unexpected checkpoint/report destination")
    probe = run/"storage_probe.json"
    probe.write_text(json.dumps({"hosted_cuda": True, "gpu": torch.cuda.get_device_name(0)}))
    assert json.loads(probe.read_text())["hosted_cuda"]
    os.chdir(project)
    os.environ["SOURCE_GIT_COMMIT"] = manifest["source_commit"]
    subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm>=4.6,<5",
                    "pandas", "pyyaml", "scikit-learn"], check=True)
    state = {"status": "starting", "source_commit": manifest["source_commit"],
             "bundle_sha256": os.environ["MULTISOURCE_BUNDLE_SHA256"],
             "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
             if "__file__" in globals() else "CLI_executed_source",
             "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    state_path.write_text(json.dumps(state, indent=2))
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(20):
            print("[Colab] Multisource worker active; checkpoints on Drive.", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with (project/"reports/multisource_neural_training.log").open("w") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "scripts.train_multisource_neural"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            state.update(status="running", remote_training_pid=child.pid)
            state_path.write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = child.wait()
        state.update(status="complete" if status == 0 else "failed", exit_code=status)
        state_path.write_text(json.dumps(state, indent=2))
        with tarfile.open(run/"multisource_neural_results.tar.gz", "w:gz") as archive:
            for folder in ("artifacts/multisource_neural", "reports"):
                archive.add(project/folder, arcname=folder)
        if status:
            raise RuntimeError("Hosted worker failed; inspect saved log before any restart")
    finally:
        finished.set()


if __name__ == "__main__":
    main()
