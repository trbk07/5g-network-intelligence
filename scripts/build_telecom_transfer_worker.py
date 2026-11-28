"""Package public Telecom prefixes/code only; reuse already-hosted original inputs."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from scripts.configure_telecom_transfer import ARMS


def main():
    root = Path.cwd()
    stage = root/".cache/telecom-transfer-source"
    worker = root/".cache/kaggle-telecom-transfer"
    stage.mkdir(parents=True, exist_ok=True)
    worker.mkdir(parents=True, exist_ok=True)
    report_path = Path("reports/telecom_italia_preparation.json")
    report = json.loads(report_path.read_text())
    manifest_path = Path(report["processed_path"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report["processed_sha256"]:
        raise ValueError("Audited Telecom manifest differs")
    if not report["training_ready"] or report["license"] != "ODbL-1.0":
        raise ValueError("Audited licensed source required")
    shards = json.loads(manifest_path.read_text())["shards"]
    files = {p.as_posix(): p for p in Path("src").rglob("*.py")}
    for name in ("__init__.py", "configure_telecom_transfer.py", "train_telecom_transfer.py",
                 "train_cell_multisource.py", "train_cell_attention.py", "train_multisource_neural.py",
                 "train_temporal_mlp.py", "train_temporal_linear.py"):
        files["scripts/"+name] = Path("scripts")/name
    for name in ("data.yaml", "evaluation.yaml", "cell_multisource_v2.yaml",
                 *("telecom_transfer_"+arm+".yaml" for arm, _, _ in ARMS)):
        files["configs/"+name] = Path("configs")/name
    files[report_path.as_posix()] = report_path
    files[manifest_path.as_posix()] = manifest_path
    for shard in shards:
        path = Path(shard["processed_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["processed_sha256"]:
            raise ValueError("Prepared Telecom shard differs")
        files[path.as_posix()] = path
    files["LICENSE"] = Path("LICENSE")
    if any(name.startswith("data/") and not name.startswith("data/processed/telecom_italia_prefix/") for name in files):
        raise ValueError("Supplement must never contain original beam data or unrelated collections")
    source_commit = subprocess.check_output(["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"source_commit": source_commit, "original_beam_payload_included": False,
                "dataset_license": "ODbL-1.0", "code_license": "MIT", "files": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    raw = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(raw).hexdigest()
    bundle = stage/"telecom_transfer_supplement.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("telecom_transfer_manifest.json", raw)
        archive.writestr("DATA_LICENSE_NOTICE.txt", "Telecom Italia database: ODbL-1.0. Attribution: from BigDataChallenge contest; "
                         "http://www.telecomitalia.com/tit/en/bigdatachallenge.html\n"
                         "Publisher conversion: https://zenodo.org/records/22721975\nCode is separately MIT; see LICENSE.\n")
    metadata = {"id": "truongbaokien/5g-ni-telecom-prefix-transfer-inputs", "title": "5G NI Telecom Prefix Transfer Inputs",
                "licenses": [{"name": "ODbL-1.0"}]}
    (stage/"dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    (stage/"README.md").write_text("# Private Telecom prefix supplement\n\nPublic licensed ODbL-1.0 source prefixes and MIT code. "
        "No original GOTS-F beam payload or checkpoint. Attribute from BigDataChallenge contest. "
        "See DATA_LICENSE_NOTICE.txt inside the archive. Existing approved Kaggle input supplies original research data separately.\n")
    base_sha = "3069c57c9f64f809464ead2fae973fbaa35a86c3a46884e15cc693354eecab46"
    sections = [
        ("markdown", "# 5G Network Intelligence — Telecom Data and Pretraining Budget Ablation\n\n"
         "## Hypothesis\nCan 12 million audited mobile-traffic prefix hours materially improve DLPRB forecasts? "
         "Three predeclared arms isolate additional data and tripled source-pretraining compute. "
         "Prior four-collection transfer failed the material-gain gate; more bytes are not a quality claim.\n"),
        ("code", "# Review Run All is safe. The dedicated hosted worker enables this flag.\nRUN_TRAINING = False\n"),
        ("markdown", "## Temporal protocol and units\nTwo reused development folds: 335/383/443 and 395/443/503 fit/stop/score. "
         "Context 168h, horizons 1/6/24h, all 2,880 beams. Original hours 504 onward remain excluded. "
         "MAE uses DLPRB usage units; WAPE is total absolute error divided by observed absolute load. "
         "Unverified utilization-percent units must not be claimed.\n"),
        ("code", "from pathlib import Path, PurePosixPath\nimport os, sys, json, hashlib, shutil\n"
         "PROJECT = Path.cwd()\nif not (PROJECT/'configs').is_dir():\n    PROJECT = PROJECT.parent\n"
         "def verified_input(filename, digest):\n"
         "    matches = [p for p in Path('/kaggle/input').rglob(filename)\n"
         "               if hashlib.sha256(p.read_bytes()).hexdigest() == digest]\n"
         "    if len(matches) != 1:\n        raise ValueError('Exactly one pinned input manifest required')\n"
         "    source = matches[0]\n    manifest = json.loads(source.read_text())\n"
         "    for name, expected in manifest['files'].items():\n"
         "        relative = PurePosixPath(name)\n"
         "        if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\\\' in name:\n"
         "            raise ValueError('Unsafe input path')\n"
         "        path = source.parent/name\n"
         "        if not path.resolve().is_relative_to(source.parent.resolve()) or path.is_symlink():\n"
         "            raise ValueError('Input path escapes manifest root')\n"
         "        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:\n"
         "            raise ValueError('Input checksum differs: '+name)\n"
         "    return source.parent, manifest\n"),
        ("markdown", "## Input provenance\nReuse the already-approved hosted original dataset; upload only public licensed Telecom prefixes and new code. "
         "Verify both manifests before importing code. The collection remains one sampling domain across shards. "
         "Whole constant/duplicate source segments are excluded; original zeros and scoring masks are unchanged. "
         "ODbL attribution/share-alike obligations apply to databases; code retains MIT.\n"),
        ("code", "if RUN_TRAINING:\n    assert Path('/kaggle/working').is_dir(), 'Hosted Kaggle worker only.'\n"
         f"    base_root, base = verified_input('source_manifest.json', {base_sha!r})\n"
         f"    extra_root, extra = verified_input('telecom_transfer_manifest.json', {digest!r})\n"
         "    assert not extra['original_beam_payload_included']\n"
         "    PROJECT = Path('/kaggle/working/telecom_transfer')\n    PROJECT.mkdir(exist_ok=True)\n"
         "    # Only reuse base data/reports; import all code from the new pinned supplement.\n"
         "    for source_root, manifest, base_only in [(base_root, base, True), (extra_root, extra, False)]:\n"
         "        for name in manifest['files']:\n"
         "            if base_only and not name.startswith(('data/', 'reports/')):\n                continue\n"
         "            target = PROJECT/name\n            target.parent.mkdir(parents=True, exist_ok=True)\n"
         "            shutil.copyfile(source_root/name, target)\n"
         "    os.chdir(PROJECT)\n    sys.path.insert(0, str(PROJECT))\n"
         "    os.environ['SOURCE_GIT_COMMIT'] = extra['source_commit']\n"
         "    print('Verified source:', extra['source_commit'])\n"),
        ("markdown", "## Three arms and causal preprocessing\n"
         "Telecom matched: 20×65,536 windows (5,120 updates), compared with the completed existing-data control. "
         "Existing extended and Telecom extended: each 60×65,536 windows (15,360 updates). "
         "Each collection has equal aggregate sampling weight. Use past-context normalization and transfer only the temporal encoder. "
         "Discard source output heads; learn physical DLPRB on original fit prefixes. Target fine-tuning remains seed 17, max 100 epochs, patience 12.\n"),
        ("code", "import yaml\nfrom IPython.display import display\n"
         "for arm in ['telecom_matched', 'existing_extended', 'telecom_extended']:\n"
         "    path = PROJECT/'configs'/f'telecom_transfer_{arm}.yaml'\n"
         "    if path.exists():\n        display(yaml.safe_load(path.read_text(encoding='utf-8')))\n"),
        ("markdown", "## Hosted training and debugging\nRun the fixed arm order, never select source checkpoints using scoring. "
         "Keep cold states, source checkpoints, optimizer/RNG and all results in Kaggle Output. "
         "Source losses are dimensionless; target losses use physical units despite historical `scaled` log names.\n"),
        ("code", "if RUN_TRAINING:\n    import subprocess\n"
         "    (PROJECT/'reports').mkdir(exist_ok=True)\n"
         "    with (PROJECT/'reports/telecom_transfer_training.log').open('w') as log:\n"
         "        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_telecom_transfer'],\n"
         "            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
         "        for line in process.stdout:\n            print(line, end='', flush=True)\n"
         "            log.write(line)\n            log.flush()\n        TRAIN_EXIT = process.wait()\n"
         "    if TRAIN_EXIT:\n        raise RuntimeError('Inspect saved log/checkpoints; do not silently repeat fitting.')\n"
         "else:\n    print('Review mode: no fitting.')\n"),
        ("markdown", "## Evaluation, acceptance and limitations\nReconstruct full-population MAE/WAPE, verify cold states, encoder isolation, "
         "optimizer budgets, stopping choices and numerical parity. At least 2% mean MAE gain with no horizon regression is only an initial research gate; "
         "5% and robustness/uncertainty checks are required for a material improvement claim. "
         "Partial source overlap and GOTS-F calendar identity are not certified absent. "
         "Repeated development, unknown Vigo scaling and correlated domains limit generalization claims. No holdout reuse or automatic production promotion.\n"),
        ("code", "# Review scores only; independent verification is a separate acceptance step.\n"
         "import pandas as pd\nfor arm in ['telecom_matched', 'existing_extended', 'telecom_extended']:\n"
         "    path = PROJECT/'reports'/f'telecom_transfer_{arm}_multi_source_pretrain_development.csv'\n"
         "    if path.exists():\n        frame = pd.read_csv(path)\n"
         "        frame['WAPE (%)'] = 100*frame['wape']\n        print(arm)\n        display(frame)\n"),
    ]
    cells = []
    for kind, source in sections:
        cell = {"cell_type": kind, "id": f"telecom-{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = "41_telecom_data_and_pretraining_budget_ablation.ipynb"
    review = Path("notebooks")/name
    if review.exists():
        raise FileExistsError("Preserve existing research guide")
    review.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    notebook["cells"][1]["source"] = ["# Dedicated hosted worker; repository review defaults to False.\nRUN_TRAINING = True\n"]
    export = """if RUN_TRAINING:
    import tarfile
    output = Path('/kaggle/working/telecom_transfer_results.tar.gz')
    with tarfile.open(output, 'w:gz') as archive:
        archive.add(PROJECT/'artifacts/telecom_transfer', arcname='artifacts/telecom_transfer')
        for path in (PROJECT/'reports').glob('telecom_transfer*'):
            archive.add(path, arcname='reports/'+path.name)
    print('Completed outputs:', output)
"""
    notebook["cells"].append({"cell_type": "code", "id": "export-results", "metadata": {},
        "source": export.splitlines(keepends=True), "execution_count": None, "outputs": []})
    (worker/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (worker/"kernel-metadata.json").write_text(json.dumps({"id": "truongbaokien/5g-ni-telecom-data-and-compute-ablation",
        "title": "5G NI Telecom Data and Pretraining Budget Ablation", "code_file": name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["truongbaokien/5g-ni-cell-multisource-v2-inputs", metadata["id"]],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    transport = {"source_commit": source_commit, "manifest_sha256": digest, "files": len(files),
                 "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(), "bytes": bundle.stat().st_size,
                 "original_beam_payload_included": False, "training_performed_locally": False}
    (stage/"transport.json").write_text(json.dumps(transport, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == "__main__":
    main()
