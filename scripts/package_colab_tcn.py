"""Package explicit TCN source/config/panel files for hosted Colab, without fitting."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    root=Path.cwd()
    files=list(root.glob("src/**/*.py"))
    files.extend(root/"scripts"/name for name in (
        "__init__.py","train_temporal_tcn.py","train_temporal_mlp.py","train_temporal_linear.py"))
    files.extend(root/"configs"/name for name in ("data.yaml","evaluation.yaml","temporal_tcn.yaml"))
    files.extend(root/"data/processed"/name for name in ("panel.npz","panel_meta.json","split.json"))
    files.append(root/"LICENSE")
    if any(not p.is_file() for p in files):
        raise ValueError("Complete canonical panel and named source files are required")
    manifest={"source_commit":subprocess.check_output(
        ["git","-c",f"safe.directory={root.as_posix()}","rev-parse","HEAD"],text=True).strip(),
        "files":{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "training_local":False}
    output=root/".cache/colab-tcn"
    output.mkdir(parents=True,exist_ok=True)
    archive_path=output/"tcn_source.zip"
    with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,p.relative_to(root).as_posix())
        archive.writestr("colab_source_manifest.json",json.dumps(manifest,indent=2))
    evidence={"source_commit":manifest["source_commit"],"files":len(files),
        "bundle_sha256":hashlib.sha256(archive_path.read_bytes()).hexdigest(),"bytes":archive_path.stat().st_size}
    (output/"transport.json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    main()
