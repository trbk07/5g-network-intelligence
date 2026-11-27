"""Readable cloud-only clipped/hurdle blend guide and private CPU worker."""

import hashlib
import json
import zipfile
from pathlib import Path
import subprocess


def main():
    root = Path.cwd()
    package = root/".cache/clipped-blend-source"
    package.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(root).as_posix(): p for p in root.glob("src/**/*.py")}
    for name in ("__init__.py", "select_development_blend.py", "select_clipped_blend.py"):
        files[f"scripts/{name}"] = root/"scripts"/name
    files["configs/clipped_blend.yaml"] = root/"configs/clipped_blend.yaml"
    files["LICENSE"] = root/"LICENSE"
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    with zipfile.ZipFile(package/"clipped_blend_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("clipped_blend_source_manifest.json", manifest_bytes)
    (package/"dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-clipped-blend-source", "title": "5G NI Clipped Blend Source",
        "licenses": [{"name": "MIT"}]}, indent=2))
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "id": f"clipped-blend-{len(cells):02d}",
                "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# 5G Network Intelligence — Clipped Forecast Blending\n\n"
        "## Hypothesis and evidence\n\nNonnegative direct training reduces zero-target error "
        "but often increases positive-target error. Its residual correlation with hurdle "
        "is approximately 0.93–0.95, which does not prove blending will help. Test a bounded "
        "convex blend and retain weight zero. No base model is retrained.\n")
    add("code", "# Choosing coefficients is calibration: hosted execution only.\nRUN_SELECTION = False\n")
    add("markdown", "## Temporal protocol and acceptance\n\nSelect each horizon's weight "
        "from 0/0.05/0.10/0.20 on inner1 only, ties favor smaller weights. Persist all choices "
        "before reading inner2 numeric forecasts. Apply unchanged on inner2. Require no "
        "horizon regression against hurdle or the previously confirmed MLP-B/hurdle blend, "
        "and at least 0.1% relative MAE gain at one horizon with nonzero weight. Do not adjust "
        "weights after confirmation. Both folds are reused development evidence. Validation "
        "504–671, final holdout and production are excluded.\n")
    add("code", "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\n"
        "# Review saved outputs without fitting or selection.\n"
        "PROTOCOL_FILE = Path('configs/clipped_blend.yaml')\n"
        "if PROTOCOL_FILE.exists():\n    import yaml\n    print(yaml.safe_load(PROTOCOL_FILE.read_text()))\n")
    add("markdown", "## Immutable sources and preprocessing\n\nUse saved direct-clipped "
        "forecasts, original LightGBM hurdle forecasts and the previous MLP blend confirmation. "
        "Source manifest hashes are frozen in the config. Every source file is verified; "
        "actual/origin/beam axes must match exactly. No new preprocessing, raw panel access "
        "or prediction-threshold fitting is needed.\n")
    setup = (
        "if RUN_SELECTION:\n"
        "    import os, sys, json, hashlib, shutil\n"
        "    if not Path('/kaggle/working').is_dir():\n        raise RuntimeError('Hosted Kaggle required')\n"
        "    paths = list(Path('/kaggle/input').rglob('clipped_blend_source_manifest.json'))\n"
        "    if len(paths) != 1:\n        raise RuntimeError('Exactly one source package required')\n"
        f"    if hashlib.sha256(paths[0].read_bytes()).hexdigest() != {manifest_sha!r}:\n"
        "        raise ValueError('Pinned source manifest differs')\n"
        "    manifest = json.loads(paths[0].read_text())\n    source = paths[0].parent\n"
        "    PROJECT = Path('/kaggle/working/clipped_blend')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    for name, digest in manifest['files'].items():\n"
        "        if not (source/name).resolve().is_relative_to(source.resolve()):\n"
        "            raise ValueError('Source path escapes package')\n"
        "        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:\n"
        "            raise ValueError('Source checksum differs')\n"
        "    for name in manifest['files']:\n"
        "        target = PROJECT/name\n        target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copyfile(source/name, target)\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0, str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']\n"
        "    import yaml\n    protocol = yaml.safe_load(Path('configs/clipped_blend.yaml').read_text())\n"
        "    def attached_root(basename, digest):\n"
        "        matches = [p.parent for p in Path('/kaggle/input').rglob('manifest.json')\n"
        "                   if p.parent.name == basename and hashlib.sha256(p.read_bytes()).hexdigest() == digest]\n"
        "        if not matches:\n            raise RuntimeError('Missing pinned completed source: '+basename)\n"
        "        return sorted(matches)[0]\n"
        "    CLIPPED = attached_root('direct_clipped', protocol['clipped_manifest_sha256'])\n"
        "    HURDLE = attached_root('hurdle_research', protocol['hurdle_manifest_sha256'])\n"
        "    CURRENT = attached_root('mlp_blend', protocol['current_manifest_sha256'])\n")
    # Execution-only environment setup is kept out of the reader's main review flow.
    add("markdown", "## Cloud execution controls\n\nThe dedicated worker attaches three "
        "completed private source jobs. Its setup verifies the source package and locates "
        "pinned artifact roots. Review Run All skips all calibration. Checkpoints remain "
        "immutable and new selection outputs stay in Kaggle Output.\n")
    add("code", "if RUN_SELECTION:\n    import subprocess, sys, tarfile\n"
        "    command = [sys.executable, '-u', '-m', 'scripts.select_clipped_blend',\n"
        "               '--clipped-root', str(CLIPPED), '--hurdle-root', str(HURDLE), '--current-root', str(CURRENT)]\n"
        "    subprocess.run(command, check=True)\n"
        "    with tarfile.open('/kaggle/working/clipped_blend_results.tar.gz', 'w:gz') as archive:\n"
        "        for folder in ('artifacts/clipped_blend', 'reports'):\n"
        "            archive.add(folder, arcname=folder)\n"
        "else:\n    print('Review mode: coefficient selection skipped.')\n")
    add("markdown", "## Evaluation and limitations\n\nReport every horizon, zero-weight "
        "choices and regressions. MAE is in original DLPRB units; WAPE is sum absolute error "
        "/ sum absolute actual. A development pass does not authorize validation refitting "
        "or establish unseen-test superiority. A strict-gate failure ends this recipe.\n")
    add("code", "for name in ('clipped_blend_selection_grid.csv', 'clipped_blend_confirmation.csv', 'clipped_blend_current_control.csv'):\n"
        "    path = Path('reports')/name\n    if path.exists():\n        display(pd.read_csv(path))\n")
    if Path("reports/clipped_blend_verification.json").exists():
        add("markdown", "## Completed and verified result\n\nThe selected clipped weight "
            "was 0.20 at every horizon. Saved forecasts replay exactly. Hurdle-only gains "
            "confirm, but the strict current-control gate fails: +24h regresses and the "
            "+1h/+6h gains remain below 0.1%. Reject this recipe; no coefficient changes, "
            "validation refit or production promotion. Evidence: `reports/clipped_blend_analysis.md`.\n")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    name = "30_clipped_forecast_blending.ipynb"
    Path("notebooks", name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    worker = json.loads(json.dumps(notebook))
    worker["cells"][1]["source"] = ["# Dedicated hosted calibration worker.\nRUN_SELECTION = True\n"]
    worker["cells"].insert(4, {"cell_type": "code", "id": "verified-cloud-sources", "metadata": {},
        "source": setup.splitlines(keepends=True), "outputs": [], "execution_count": None})
    stage = root/".cache/kaggle-clipped-blend"
    stage.mkdir(parents=True, exist_ok=True)
    (stage/name).write_text(json.dumps(worker, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-clipped-forecast-blending", "title": "5G NI Clipped Forecast Blending",
        "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "kernel_sources": ["truongbaokien/5g-network-intelligence-zero-inflated-forecasting",
                           "truongbaokien/5g-ni-mae-aligned-forecasting",
                           "truongbaokien/5g-network-intelligence-mlp-blend"],
        "dataset_sources": ["truongbaokien/5g-ni-clipped-blend-source"], "competition_sources": []}, indent=2))
    print("Staged reader, private CPU worker and source package; no selection performed locally")


if __name__ == "__main__":
    main()
