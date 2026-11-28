"""Small verified inference-only worker reusing original private inputs/checkpoints."""

import argparse
import hashlib
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multisource", action="store_true")
    args = parser.parse_args()
    study = "cell_multisource" if args.multisource else "cell_attention"
    source = Path(".cache/cell-multi-precision-source" if args.multisource else ".cache/cell-precision-source")
    stage = Path(".cache/kaggle-cell-multi-precision" if args.multisource else ".cache/kaggle-cell-precision")
    source.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    files = {p.as_posix(): p for p in Path("src").rglob("*.py")}
    for name in ("__init__.py", "diagnose_cell_precision.py", "train_cell_attention.py",
                 "train_multisource_neural.py", "train_temporal_mlp.py", "train_temporal_linear.py"):
        files["scripts/"+name] = Path("scripts")/name
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "files": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}}
    raw = json.dumps(manifest, indent=2).encode()
    code_sha = hashlib.sha256(raw).hexdigest()
    trained_sha = hashlib.sha256(Path(f"artifacts/{study}/manifest.json").read_bytes()).hexdigest()
    base_sha = json.loads(Path(f"reports/{study}_cloud_submission.json").read_text())["manifest_sha256"]
    with zipfile.ZipFile(source/"precision_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("precision_source_manifest.json", raw)
    (source/"LICENSE").write_bytes(Path("LICENSE").read_bytes())
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-cell-multi-precision-code" if args.multisource else "truongbaokien/5g-ni-cell-precision-code",
        "title": "5G NI Cell Multi Precision Code" if args.multisource else "5G NI Cell Precision Code",
        "licenses": [{"name": "MIT"}]}, indent=2))
    setup = f'''# Pin every code/input/checkpoint member before importing modules.
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, sys
def verified_manifest(filename, expected):
    matches = [p for p in Path('/kaggle/input').rglob(filename)
               if hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    if len(matches) != 1:
        raise ValueError('Exactly one pinned manifest required: ' + filename)
    path = matches[0]
    manifest = json.loads(path.read_text())
    for name, digest in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or chr(92) in name:
            raise ValueError('Unsafe member')
        original = path.parent/name
        if not original.resolve().is_relative_to(path.parent.resolve()) or hashlib.sha256(original.read_bytes()).hexdigest() != digest:
            raise ValueError('Member identity differs: ' + name)
    return path, manifest
code_path, code = verified_manifest('precision_source_manifest.json', {code_sha!r})
base_path, base = verified_manifest('source_manifest.json', {base_sha!r})
trained_path, trained = verified_manifest('manifest.json', {trained_sha!r})
PROJECT = Path('/kaggle/working/cell_precision')
PROJECT.mkdir(exist_ok=True)
for folder in ('data', 'configs'):
    shutil.copytree(base_path.parent/folder, PROJECT/folder, dirs_exist_ok=True)
for name in code['files']:
    target = PROJECT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(code_path.parent/name, target)
TRAINED_ROOT = trained_path.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = code['source_commit']
print('Verified inference code:', code['source_commit'])
'''
    setup = "if RUN_REPLAY:\n"+textwrap.indent(setup, "    ")+"else:\n    from pathlib import Path\n    PROJECT = Path.cwd()\n    print('Review mode: attached-input verification skipped.')\n"
    run = ("# Frozen checkpoints only: no training or checkpoint selection.\n"
        "if RUN_REPLAY:\n    import subprocess, sys\n"
        "    (PROJECT/'reports').mkdir(exist_ok=True)\n"
        "    with (PROJECT/'reports/cell_attention_precision_inference.log').open('w') as log:\n"
        "        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.diagnose_cell_precision',\n"
        "            '--trained-root', str(TRAINED_ROOT)], stdout=subprocess.PIPE,\n"
        "            stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "        for line in process.stdout:\n            print(line, end='', flush=True)\n"
        "            log.write(line)\n            log.flush()\n        REPLAY_EXIT = process.wait()\n"
        "else:\n    print('Review only: inference skipped.')\n")
    if args.multisource:
        run = ("# Replay the two frozen cases; never fit or reselect checkpoints.\n"
            "if RUN_REPLAY:\n    import subprocess, sys\n"
            "    (PROJECT/'reports').mkdir(exist_ok=True)\n    REPLAY_EXIT = 0\n"
            "    with (PROJECT/'reports/cell_multisource_precision_inference.log').open('w') as log:\n"
            "        for case in ('target_self_pretrain', 'multi_source_pretrain'):\n"
            "            process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.diagnose_cell_precision',\n"
            "                '--trained-root', str(TRAINED_ROOT/case), '--case', case],\n"
            "                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "            for line in process.stdout:\n                print(line, end='', flush=True)\n"
            "                log.write(line)\n                log.flush()\n"
            "            REPLAY_EXIT = process.wait()\n            if REPLAY_EXIT:\n                break\n"
            "    if not REPLAY_EXIT:\n        output_root = PROJECT/'artifacts/cell_multisource_precision'\n"
            "        files = {p.relative_to(output_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()\n"
            "                 for p in output_root.rglob('*') if p.is_file() and p != output_root/'manifest.json'}\n"
            "        (output_root/'manifest.json').write_text(json.dumps({'files':files}, indent=2))\n"
            "else:\n    print('Review mode: replay skipped.')\n")
    sections = [("markdown", "# 5G Network Intelligence — Cell Attention Numerical Consistency\n\n"
        "## Question\nDoes the GPU attention inference fastpath explain saved forecast "
        "differences versus CPU? Test both backends with fixed checkpoints and origins.\n"),
        ("markdown", "## Protocol\nNo fit, calibration, checkpoint reselection or holdout. "
        "Keep original best weights. Compare three cells at two predeclared origins "
        "per fold. Absolute tolerance remains 1e-4 DLPRB units. Reproduce original GPU "
        "forecasts first, then require strict-math CPU/GPU parity before full replay. "
        "Backend choice is prescribed for consistency, never selected by MAE/WAPE.\n"),
        ("code", "RUN_REPLAY = False\n"), ("code", setup),
        ("markdown", "## Backend diagnosis and strict forecast replay\nThe implementation "
        "is `scripts/diagnose_cell_precision.py`. Preserve original artifacts and export "
        "new forecasts separately. Training results remain historical snapshots.\n"), ("code", run),
        ("markdown", "## Limitations\nAn inference backend fix is not evidence of improved "
        "model quality. Independently reconstruct metrics and parity after export. "
        "Failure preserves diagnostics and does not relax the tolerance.\n")]
    cells = []
    for kind, text in sections:
        cell = {"cell_type": kind, "id": f"precision-{len(cells):02d}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    name = "36_multi_corpus_numerical_consistency.ipynb" if args.multisource else "35_cell_attention_numerical_consistency.ipynb"
    if args.multisource:
        notebook["cells"][0]["source"] = ["# 5G Network Intelligence — Multi-Corpus Numerical Consistency\n\n",
            "Inference-only replay of target-self and expanded-data cases, using unchanged weights/epochs and the same strict attention policy.\n"]
    Path("notebooks", name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    notebook["cells"][2]["source"] = ["# Dedicated hosted inference worker.\nRUN_REPLAY = True\n"]
    export = ("# Export diagnostics even if inference consistency failed.\nimport tarfile\n"
        "output = Path('/kaggle/working/cell_attention_precision_results.tar.gz')\n"
        "with tarfile.open(output, 'w:gz') as archive:\n"
        "    for folder in ('artifacts/cell_attention_precision', 'reports'):\n"
        "        if (PROJECT/folder).exists():\n            archive.add(PROJECT/folder, arcname=folder)\n"
        "print(output)\nif REPLAY_EXIT:\n    raise RuntimeError('Inspect exported backend diagnosis; no training was run')\n")
    if args.multisource:
        export = export.replace("cell_attention_precision", "cell_multisource_precision")
    notebook["cells"].append({"cell_type": "code", "id": "export-precision", "metadata": {},
        "source": export.splitlines(keepends=True), "execution_count": None, "outputs": []})
    (stage/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-cell-multi-precision-replay" if args.multisource else "truongbaokien/5g-ni-cell-attention-precision-replay",
        "title": "5G NI Cell Multi Precision Replay" if args.multisource else "5G NI Cell Attention Precision Replay",
        "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": (["truongbaokien/5g-ni-cell-multisource-inputs", "truongbaokien/5g-ni-cell-multi-precision-code"]
                            if args.multisource else ["truongbaokien/5g-ni-cell-attention-inputs", "truongbaokien/5g-ni-cell-precision-code"]),
        "kernel_sources": ["truongbaokien/5g-ni-multi-corpus-spatial-encoder-pretraining" if args.multisource else
                           "truongbaokien/5g-ni-cell-spatial-temporal-attention"], "competition_sources": []}, indent=2))
    print(json.dumps({"code_manifest_sha256": code_sha, "trained_manifest_sha256": trained_sha,
                      "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
