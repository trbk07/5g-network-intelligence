"""Bundle code/config/public URLs only; never export local beam data to Colab."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from src.data.panel import load_panel
from src.utils.config import load_config


def main():
    root = Path.cwd()
    stage = root/".cache/colab-seeded-cell"
    stage.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path("reports/data_manifest.json").read_text())
    if source["license"].lower() != "mit" or source["repo_id"] != "netop/gotsf-ds":
        raise ValueError("Expected the public MIT publisher source")
    files = {p.as_posix(): p for p in Path("src").rglob("*.py")}
    for name in ("__init__.py", "run_colab_seeded_cell.py", "run_colab_tcn.py", "train_cell_attention_seed29.py",
                 "train_cell_attention.py", "train_multisource_neural.py", "train_temporal_mlp.py", "train_temporal_linear.py"):
        files["scripts/"+name] = Path("scripts")/name
    for name in ("data.yaml", "evaluation.yaml", "cell_attention_seed29.yaml"):
        files["configs/"+name] = Path("configs")/name
    files["LICENSE"] = Path("LICENSE")
    cfg = load_config("configs/data.yaml")
    panel = load_panel(Path(cfg["processed_dir"])/"panel.npz")
    public = {"repo_id": source["repo_id"], "revision": source["revision"], "license": "MIT",
              "frozen_split": json.loads(Path("data/processed/split.json").read_text()),
              "prefix_values_sha256": hashlib.sha256(panel.values[:504].tobytes()).hexdigest(), "train_files": {}}
    for kpi in ("DLPRB", "DLThpVol", "DLThpTime", "MR_number"):
        name = f"data/train/{kpi}_train_0w-5w.csv"
        public["train_files"][kpi] = {"path": name, "sha256": source["checksums"][name],
            "url": f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{name}?download=true"}
    public_path = stage/"public_sources.json"
    public_path.write_text(json.dumps(public, indent=2))
    files["public_sources.json"] = public_path
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "files": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()},
        "local_beam_payload_included": False}
    archive_path = stage/"seeded_cell_code.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    transport = {"source_commit": manifest["source_commit"], "files": len(files),
                 "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "bytes": archive_path.stat().st_size,
                 "local_beam_payload_included": False, "training_performed": False}
    (stage/"transport.json").write_text(json.dumps(transport, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == "__main__":
    main()
