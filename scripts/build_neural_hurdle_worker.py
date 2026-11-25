"""Package redacted development inputs and a readable private GPU worker."""

import hashlib
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.data.panel import load_panel, save_panel


def main(study="neural_hurdle"):
    if study not in ("neural_hurdle", "mae_aligned"):
        raise ValueError("Unknown study")
    stem = study.replace("_", "-")
    notebook_name = ("28_multivariate_zero_inflated_forecasting.ipynb" if study == "neural_hurdle"
                     else "29_mae_aligned_forecasting.ipynb")
    title = ("5G NI Multivariate Zero Inflated Forecasting" if study == "neural_hurdle"
             else "5G NI MAE Aligned Forecasting")
    slug = ("5g-ni-multivariate-zero-inflated-forecasting" if study == "neural_hurdle"
            else "5g-ni-mae-aligned-forecasting")
    root = Path.cwd()
    source = root/f".cache/{stem}-source"
    source.mkdir(parents=True, exist_ok=True)
    panel = load_panel("data/processed/panel.npz")
    values = np.full(panel.values.shape, np.nan, dtype=panel.values.dtype)
    values[:504] = panel.values[:504]
    stage_panel = root/f".cache/{stem}-panel/panel.npz"
    save_panel(replace(panel, values=values), stage_panel)
    files = {p.relative_to(root).as_posix(): p for p in root.glob("src/**/*.py")}
    for name in ("__init__.py", f"train_{study}.py", "train_temporal_mlp.py",
                 "train_temporal_linear.py", "train_multisource_neural.py"):
        files[f"scripts/{name}"] = root/"scripts"/name
    for name in ("data.yaml", "evaluation.yaml", f"{study}.yaml"):
        files[f"configs/{name}"] = root/"configs"/name
    files.update({"data/processed/panel.npz": stage_panel,
                  "data/processed/panel_meta.json": stage_panel.with_name("panel_meta.json"),
                  "data/processed/split.json": root/"data/processed/split.json",
                  "LICENSE": root/"LICENSE"})
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "original_values_available_through": 503,
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    bundle = source/f"{study}_source.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("source_manifest.json", manifest_bytes)
    transport = {"source_commit": manifest["source_commit"], "files": len(files),
                 "manifest_sha256": digest, "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                 "bytes": bundle.stat().st_size, "original_values_504_onward_redacted": True,
                 "training_performed": False}
    (source/"transport.json").write_text(json.dumps(transport, indent=2))
    (source/"DATA_SOURCES.md").write_text("# Attribution\n\nNETOP GOTS-F: "
        "https://github.com/netop/gotsf-ds (MIT). Project code: MIT, LICENSE in bundle. "
        "Original values from hour 504 onward have been replaced by NaN. "
        "This job does not include external transfer datasets.\n")
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id": f"truongbaokien/5g-ni-{stem}-inputs", "title": f"5G NI {stem.title()} Inputs",
        "licenses": [{"name": "MIT"}]}, indent=2))
    notebook = json.loads(Path("notebooks", notebook_name).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "RUN_TRAINING = False" in text:
            cell["source"] = ["# Dedicated hosted GPU worker; the repository review defaults to False.\nRUN_TRAINING = True\n"]
    setup = (
        "# Verify every attached source/data member before importing project code.\n"
        "from pathlib import Path, PurePosixPath\nimport os, sys, json, hashlib, shutil\n"
        "candidates = list(Path('/kaggle/input').rglob('source_manifest.json'))\n"
        "if len(candidates) != 1:\n    raise RuntimeError('Exactly one source manifest required')\n"
        "manifest_path = candidates[0]\n"
        f"if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != {digest!r}:\n"
        "    raise ValueError('Pinned manifest checksum differs')\n"
        "manifest = json.loads(manifest_path.read_text())\ninput_root = manifest_path.parent\n"
        f"project = Path('/kaggle/working/{study}')\nproject.mkdir(exist_ok=True)\n"
        "for name, expected in manifest['files'].items():\n"
        "    relative = PurePosixPath(name)\n"
        "    if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\\\' in name:\n"
        "        raise ValueError('Unsafe source path')\n"
        "    original = input_root / name\n"
        "    if not original.resolve().is_relative_to(input_root.resolve()):\n"
        "        raise ValueError('Source path escapes input root')\n"
        "    if hashlib.sha256(original.read_bytes()).hexdigest() != expected:\n"
        "        raise ValueError('Source checksum differs: ' + name)\n"
        "for name in manifest['files']:\n"
        "    target = project / name\n    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(input_root / name, target)\n"
        "os.chdir(project)\nsys.path.insert(0, str(project))\n"
        "os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']\n"
        "print('Verified source:', manifest['source_commit'])\n")
    notebook["cells"].insert(4, {"cell_type": "code", "id": "verified-worker-inputs",
        "metadata": {}, "source": setup.splitlines(keepends=True), "execution_count": None, "outputs": []})
    export = ("# Preserve completed outputs for independent CPU replay.\nimport tarfile\n"
        f"output = Path('/kaggle/working/{study}_results.tar.gz')\n"
        "with tarfile.open(output, 'w:gz') as archive:\n"
        f"    for folder in ('artifacts/{study}', 'reports'):\n"
        "        archive.add(PROJECT / folder, arcname=folder)\nprint(output)\n")
    notebook["cells"].append({"cell_type": "code", "id": "export-durable-results",
        "metadata": {}, "source": export.splitlines(keepends=True), "execution_count": None, "outputs": []})
    stage = root/f".cache/kaggle-{stem}"
    stage.mkdir(parents=True, exist_ok=True)
    name = notebook_name
    (stage/name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id": f"truongbaokien/{slug}",
        "title": title, "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"truongbaokien/5g-ni-{stem}-inputs"],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["neural_hurdle", "mae_aligned"], default="neural_hurdle")
    main(parser.parse_args().study)
