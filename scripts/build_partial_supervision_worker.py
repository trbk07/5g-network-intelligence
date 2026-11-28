"""Create a code-only cloud supplement and readable partial-label experiment guide."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    stage = Path(".cache/partial-supervision-code")
    worker = Path(".cache/kaggle-partial-supervision")
    stage.mkdir(parents=True, exist_ok=True)
    worker.mkdir(parents=True, exist_ok=True)
    files = {p.as_posix(): p for p in Path("src").rglob("*.py")}
    for name in ("__init__.py", "train_partial_supervision.py", "configure_partial_supervision.py", "train_cell_attention.py",
                 "train_multisource_neural.py", "train_temporal_mlp.py", "train_temporal_linear.py"):
        files["scripts/"+name] = Path("scripts")/name
    for name in ("data.yaml", "evaluation.yaml", "cell_attention_v2.yaml", "partial_supervision_joint.yaml", "partial_supervision_available.yaml"):
        files["configs/"+name] = Path("configs")/name
    files["LICENSE"] = Path("LICENSE")
    if any(name.startswith(("data/", "artifacts/")) for name in files):
        raise ValueError("This supplement must contain code/config only")
    commit = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"source_commit": commit, "data_or_checkpoint_included": False,
                "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    raw = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(raw).hexdigest()
    archive_path = stage/"partial_supervision_code.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("partial_supervision_source_manifest.json", raw)
    dataset = "truongbaokien/5g-ni-partial-supervision-code"
    (stage/"dataset-metadata.json").write_text(json.dumps({"id": dataset,
        "title": "5G NI Partial Supervision Code", "licenses": [{"name": "MIT"}]}, indent=2))
    (stage/"README.md").write_text("# Private code-only supplement\n\nMIT code/configuration. No raw observations, forecasts or checkpoints. "
        "Reuse the already hosted original input; never train locally.\n")
    reference = json.loads(Path("notebooks/41_telecom_data_and_pretraining_budget_ablation.ipynb").read_text(encoding="utf-8"))
    setup = "".join(reference["cells"][3]["source"])
    ingress = ("if RUN_TRAINING:\n    assert Path('/kaggle/working').is_dir(), 'Hosted Kaggle only.'\n"
        "    base_root, base = verified_input('source_manifest.json',\n"
        "        '3069c57c9f64f809464ead2fae973fbaa35a86c3a46884e15cc693354eecab46')\n"
        f"    code_root, code = verified_input('partial_supervision_source_manifest.json', {digest!r})\n"
        "    assert not code['data_or_checkpoint_included']\n"
        "    PROJECT = Path('/kaggle/working/partial_supervision_study')\n    PROJECT.mkdir(exist_ok=True)\n"
        "    # Reuse only canonical original panel/split. External source corpora are unused.\n"
        "    for name in ['data/processed/panel.npz', 'data/processed/panel_meta.json', 'data/processed/split.json']:\n"
        "        target = PROJECT/name\n        target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copyfile(base_root/name, target)\n"
        "    for name in code['files']:\n        target = PROJECT/name\n"
        "        target.parent.mkdir(parents=True, exist_ok=True)\n        shutil.copyfile(code_root/name, target)\n"
        "    os.chdir(PROJECT)\n    sys.path.insert(0, str(PROJECT))\n"
        "    os.environ['SOURCE_GIT_COMMIT'] = code['source_commit']\n"
        "    print('Verified training source:', code['source_commit'])\n")
    sections = [
        ("markdown", "# 5G Network Intelligence — Available Target Supervision\n\n"
         "## Research question and hypothesis\nCan retaining valid short-horizon fit labels improve forecasts? "
         "Compare newly trained joint-label and available-label arms with the same architecture, seed and horizon-balanced "
         "training objective. The old control is not silently treated as loss-matched. "
         "Implementation tests establish causality, not an accuracy gain.\n"),
        ("code", "# Review Run All never trains. A separate hosted worker enables this flag.\nRUN_TRAINING = False\n"),
        ("markdown", "## Frozen temporal protocol\nTwo reused development folds: fit/stop/score endpoints "
         "335/383/443 and 395/443/503. Context 168h, all 2,880 beams, horizons 1/6/24h. "
         "Original hours 504 onward remain excluded. Both arms keep complete joint stopping windows and identical "
         "full per-horizon scoring populations. No final refit, holdout tuning or production promotion.\n"),
        ("code", setup),
        ("markdown", "## Input provenance\nThe new private supplement contains only MIT code/config. "
         "Reuse original observations from the already approved hosted input after manifest verification. "
         "No local data export, external pretraining corpus or checkpoint transfer is needed.\n"),
        ("code", ingress),
        ("markdown", "## Configuration\nBoth arms use seed 17, cell attention, maximum 100 epochs and patience 12. "
         "Global per-KPI medians/scales fit only the training prefix; phase4 and missing masks are retained.\n"),
        ("code", "import yaml\nimport pandas as pd\nfrom IPython.display import display\nsettings = []\n"
         "for arm in ['joint', 'available']:\n    path = PROJECT/'configs'/f'partial_supervision_{arm}.yaml'\n"
         "    if path.exists():\n        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))\n"
         "        settings.append({'arm': arm, 'seed': cfg['seed'], 'batch_size': cfg['batch_size'],\n"
         "            'max_epochs': cfg['max_epochs'], 'patience': cfg['early_stopping_patience'],\n"
         "            'training_objective': cfg['training_objective'], 'stopping': cfg['stopping_label_policy']})\n"
         "display(pd.DataFrame(settings))\n"),
        ("markdown", "## Causal labels and loss\nCrop the backing tensors before transformation/gathering. "
         "An origin+horizon beyond the fit cutoff is NaN, never an imputed target. For observed labels, average "
         "MAE within each horizon and then across supervised horizons in each batch. Use this same loss in both arms. "
         "With complete observed prefixes, available labels add 118,080 targets per fold (9.43%/6.67%). "
         "Natural window counts and optimizer updates also increase; do not call this a pure data effect without a step-matched control.\n"),
        ("code", "if RUN_TRAINING:\n    import subprocess\n    (PROJECT/'reports').mkdir(exist_ok=True)\n"
         "    with (PROJECT/'reports/partial_supervision_training.log').open('w') as log:\n"
         "        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_partial_supervision'],\n"
         "            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
         "        for line in process.stdout:\n            print(line, end='', flush=True)\n"
         "            log.write(line)\n            log.flush()\n        TRAIN_EXIT = process.wait()\n"
         "    if TRAIN_EXIT:\n        raise RuntimeError('Inspect durable logs/checkpoints before any repeat fit.')\n"
         "else:\n    print('Review mode: no fitting.')\n"),
        ("markdown", "## Debugging and independent evaluation\nSave per-horizon supervision counts, cold initialization, "
         "best/last checkpoints, optimizer/RNG and stopping history in Kaggle Output. Training logs contain a weighted "
         "average of batch objectives, not a reconstructed global macro MAE. Stopping logs use physical DLPRB MAE "
         "despite the historical `scaled` field name. Verify the loss, counts, axes, preprocessing and numerical parity "
         "before assessing quality. MAE uses DLPRB usage units; WAPE is total absolute error divided by total absolute observed load.\n"),
        ("code", "# Inspect completed results only; no new selection or calibration in review.\n"
         "for arm in ['joint', 'available']:\n    path = PROJECT/'reports'/f'partial_supervision_{arm}_development.csv'\n"
         "    if path.exists():\n        frame = pd.read_csv(path)\n        frame['WAPE (%)'] = 100*frame['wape']\n"
         "        print(arm)\n        display(frame)\n"),
        ("markdown", "## Decision and limitations\nRequire a full-population comparison with the new joint control and fixed strong blend, "
         "report every fold/horizon and negative result. A 2% mean MAE point gain without regression is an initial screen; "
         "a 5% material gain still needs robustness and uncertainty evidence. WAPE below 10% is unproven. "
         "More labels do not guarantee lower error; training exposure and per-batch loss weighting must be disclosed. "
         "This is reused development, not fresh generalization. The existing Telecom study is not changed or retrained by this guide.\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "id": f"available-target-{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = "43_available_target_supervision.ipynb"
    review = Path("notebooks")/name
    if review.exists():
        raise FileExistsError("Preserve the existing research guide")
    review.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    notebook["cells"][1]["source"] = ["# Dedicated hosted GPU training worker.\nRUN_TRAINING = True\n"]
    export = """if RUN_TRAINING:
    import tarfile
    output = Path('/kaggle/working/partial_supervision_results.tar.gz')
    with tarfile.open(output, 'w:gz') as archive:
        archive.add(PROJECT/'artifacts/partial_supervision', arcname='artifacts/partial_supervision')
        for path in (PROJECT/'reports').glob('partial_supervision*'):
            archive.add(path, arcname='reports/'+path.name)
    print('Completed results remain in Kaggle Output:', output)
"""
    notebook["cells"].append({"cell_type": "code", "id": "export-durable-results", "metadata": {},
        "source": export.splitlines(keepends=True), "execution_count": None, "outputs": []})
    (worker/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (worker/"kernel-metadata.json").write_text(json.dumps({"id": "truongbaokien/5g-ni-available-target-supervision",
        "title": "5G NI Available Target Supervision", "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["truongbaokien/5g-ni-cell-multisource-v2-inputs", dataset],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    transport = {"source_commit": commit, "manifest_sha256": digest, "files": len(files),
                 "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                 "bytes": archive_path.stat().st_size, "data_or_checkpoint_included": False, "training_performed": False}
    (stage/"transport.json").write_text(json.dumps(transport, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == "__main__":
    main()
