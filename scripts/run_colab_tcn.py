"""Hosted Colab TCN launcher. CLI supplies TCN_BUNDLE_SHA256 and TCN_RUN_NAME."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile


def verify_transport(archive_path, expected_sha256, manifest_name="colab_source_manifest.json"):
    """Validate the complete allowlisted transport before extracting anything."""
    if hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError("Transport checksum differs")
    with zipfile.ZipFile(archive_path) as archive:
        names=[]
        for member in archive.infolist():
            path=PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or ":" in member.filename
                    or "\\" in member.filename or member.is_dir()
                    or stat.S_ISLNK(member.external_attr>>16)):
                raise ValueError("Unsafe transport member")
            names.append(member.filename)
        if len(names)!=len(set(names)):
            raise ValueError("Duplicate transport members")
        manifest=json.loads(archive.read(manifest_name))
        if set(names)!=set(manifest["files"])|{manifest_name}:
            raise ValueError("Transport does not match its manifest")
        for name,digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError("Source/data checksum differs: "+name)
    return manifest


def main():
    # Refuse even preparation on the owner's local machine; no fitting can follow.
    if os.name=="nt" or "COLAB_RELEASE_TAG" not in os.environ or not Path("/content").is_dir():
        raise RuntimeError("This launcher requires hosted Colab; local training is forbidden")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Hosted CUDA is required")
    drive=Path("/content/drive/MyDrive")
    if not drive.is_dir():
        raise RuntimeError("Mount Drive before training")
    run_name=os.environ["TCN_RUN_NAME"]
    if not run_name.startswith("temporal_tcn_colab_") or not all(c.isalnum() or c in "-_" for c in run_name):
        raise ValueError("A dedicated TCN run directory name is required")
    archive_path=Path("/content/tcn_source.zip")
    manifest=verify_transport(archive_path,os.environ["TCN_BUNDLE_SHA256"])
    run=drive/"5g-network-intelligence/checkpoints"/run_name
    if (run/"runtime_state.json").exists():
        raise RuntimeError("Existing execution state; inspect before resubmitting")
    run.mkdir(parents=True,exist_ok=True)
    project=Path("/content/tcn_research")
    project.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(project)
    (project/"artifacts").mkdir(exist_ok=True)
    for link,target in ((project/"artifacts/temporal_tcn",run/"artifacts/temporal_tcn"),(project/"reports",run/"reports")):
        target.mkdir(parents=True,exist_ok=True)
        if not link.exists():
            link.symlink_to(target,target_is_directory=True)
        if link.resolve()!=target.resolve():
            raise ValueError("Unexpected output storage path")
    probe=run/"storage_probe.json"
    probe.write_text(json.dumps({"hosted_cuda":True,"gpu":torch.cuda.get_device_name(0)}))
    if not json.loads(probe.read_text())["hosted_cuda"]:
        raise RuntimeError("Drive storage verification failed")
    os.chdir(project)
    os.environ["SOURCE_GIT_COMMIT"]=manifest["source_commit"]
    subprocess.run([sys.executable,"-m","pip","install","lightgbm>=4.6,<5","pandas","pyyaml","scikit-learn"],
                   check=True,stdout=subprocess.DEVNULL)
    evidence={"runtime":"Colab hosted GPU","gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
        "source_commit":manifest["source_commit"],"source_files_verified":len(manifest["files"]),
        "transport_sha256":os.environ["TCN_BUNDLE_SHA256"],"checkpoint_storage":str(run)}
    (run/"colab_transfer_evidence.json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence),flush=True)
    state_path=run/"runtime_state.json"
    state={**evidence,"status":"starting","start_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    state_path.write_text(json.dumps(state,indent=2))
    finished=threading.Event()

    def heartbeat():
        while not finished.wait(20):
            print("[Colab] TCN process active; checkpoint directory is on Drive.",flush=True)

    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        with Path("reports/temporal_tcn_training.log").open("w") as log:
            child=subprocess.Popen([sys.executable,"-u","-m","scripts.train_temporal_tcn"],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            state.update(status="running",remote_training_pid=child.pid)
            state_path.write_text(json.dumps(state,indent=2))
            print("Hosted TCN training started; PID",child.pid,flush=True)
            for line in child.stdout:
                print(line,end="",flush=True)
                log.write(line)
                log.flush()
            status=child.wait()
        state.update(status="complete" if status==0 else "failed",exit_code=status,
            end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        state_path.write_text(json.dumps(state,indent=2))
        with tarfile.open(run/"temporal_tcn_results.tar.gz","w:gz") as archive:
            for folder in ("artifacts/temporal_tcn","reports"):
                archive.add(Path(folder).resolve(),arcname=folder)
        if status:
            raise RuntimeError("Hosted TCN failed; inspect the saved log")
        print("Hosted TCN complete; result archive is on Drive.",flush=True)
    finally:
        finished.set()


if __name__=="__main__":
    main()
