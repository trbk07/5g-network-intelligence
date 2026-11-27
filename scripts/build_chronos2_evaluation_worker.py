"""Package small evaluation code; reuse immutable base inputs and trained output."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    training_bytes = args.training_manifest.read_bytes()
    training = json.loads(training_bytes)
    if not training.get("checkpoint_selection_pending") or not training.get("files"):
        raise ValueError("A completed fixed-step training manifest is required")
    for name, digest in training["files"].items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid trained member SHA256: " + name)
    for fold in ("inner1", "inner2"):
        required = {f"{fold}/training_protocol.json"}
        for step in (100, 300, 1000):
            required.update(f"{fold}/trainer/checkpoint-{step}/{name}" for name in
                            ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                             "scheduler.pt", "rng_state.pth", "trainer_state.json"))
        if not required.issubset(training["files"]):
            raise ValueError("Training manifest lacks eligible checkpoint state for " + fold)
    training_sha = hashlib.sha256(training_bytes).hexdigest()
    submission = json.loads(Path("reports/chronos2_lora_cloud_submission.json").read_text())
    stage = Path(".cache/chronos2-evaluation-source")
    stage.mkdir(parents=True, exist_ok=True)
    files = {p.as_posix(): p for p in Path("src").rglob("*.py")}
    for name in ("__init__.py", "evaluate_chronos2_lora.py", "train_multisource_neural.py",
                 "train_temporal_mlp.py", "train_temporal_linear.py"):
        files[f"scripts/{name}"] = Path("scripts")/name
    for name in ("data.yaml", "evaluation.yaml", "chronos2_lora.yaml", "chronos2_synth.yaml"):
        files[f"configs/{name}"] = Path("configs")/name
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "training_manifest_sha256": training_sha,
        "files": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}}
    raw = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(stage/"evaluation_source.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in files.items():
            bundle.write(path, name)
        bundle.writestr("evaluation_source_manifest.json", raw)
    (stage/"dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-chronos2-evaluation-code",
        "title": "5G NI Chronos2 Evaluation Code", "licenses": [{"name": "MIT"}]}, indent=2))
    (stage/"LICENSE").write_bytes(Path("LICENSE").read_bytes())
    notebook = json.loads(Path("notebooks/32_chronos2_adaptation_evaluation.ipynb").read_text(encoding="utf-8"))
    setup = f'''# Verify code and immutable inputs before importing project modules.
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, sys
def find_manifest(filename, expected):
    matches = [p for p in Path('/kaggle/input').rglob(filename)
               if hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one checksum-matching ' + filename)
    return matches[0]
def verify_files(path):
    manifest = json.loads(path.read_text())
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or chr(92) in name:
            raise ValueError('Unsafe input path')
        original = path.parent / name
        if not original.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('Input escapes root')
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise ValueError('Input checksum differs: ' + name)
    return manifest
code_path = find_manifest('evaluation_source_manifest.json', {digest!r})
base_path = find_manifest('source_manifest.json', {submission['manifest_sha256']!r})
trained_path = find_manifest('manifest.json', {training_sha!r})
code = verify_files(code_path)
verify_files(base_path)
verify_files(trained_path)
PROJECT = Path('/kaggle/working/chronos2_evaluation')
PROJECT.mkdir(exist_ok=True)
# Reuse the already redacted panel; pretrained weights stay in read-only Input.
for folder in ('data',):
    shutil.copytree(base_path.parent / folder, PROJECT / folder, dirs_exist_ok=True)
for name in code['files']:
    target = PROJECT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(code_path.parent / name, target)
MODEL_ROOT = base_path.parent / 'pretrained/chronos2_synth'
TRAINED_ROOT = trained_path.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['SOURCE_GIT_COMMIT'] = code['source_commit']
print('Verified evaluation source:', code['source_commit'])
'''
    def cell(identifier, source):
        return {"cell_type": "code", "id": identifier, "metadata": {},
                "source": source.splitlines(keepends=True), "execution_count": None, "outputs": []}
    notebook["cells"].insert(2, cell("verified-evaluation-inputs", setup))
    notebook["cells"].insert(3, cell("pinned-evaluation-packages",
        "# Match the fine-tuning environment before fresh-subprocess inference.\n"
        "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',\n"
        "    'chronos-forecasting==2.3.2', 'transformers==4.57.6', 'peft==0.18.1'], check=True)\n"))
    for item in notebook["cells"]:
        source = "".join(item["source"])
        if "RUN_EVALUATION = False" in source:
            item["source"] = source.replace("RUN_EVALUATION = False", "RUN_EVALUATION = True").splitlines(keepends=True)
        if "subprocess.run([sys.executable, '-u'" in source:
            # Export partial inference caches even when the evaluator fails.
            item["source"] = ("# Hosted inference only; preserve its log and exit status.\n"
                "import subprocess, sys\n(PROJECT / 'reports').mkdir(exist_ok=True)\n"
                "with (PROJECT / 'reports/chronos2_lora_evaluation.log').open('w') as log:\n"
                "    process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.evaluate_chronos2_lora',\n"
                "        '--model-root', str(MODEL_ROOT), '--trained-root', str(TRAINED_ROOT)],\n"
                "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
                "    for line in process.stdout:\n        print(line, end='', flush=True)\n"
                "        log.write(line)\n        log.flush()\n    EVALUATION_EXIT = process.wait()\n").splitlines(keepends=True)
    notebook["cells"].append(cell("export-evaluation-results",
        "# Keep inference caches and reports in durable Kaggle Output.\nimport tarfile\n"
        "output = Path('/kaggle/working/chronos2_lora_evaluation_results.tar.gz')\n"
        "with tarfile.open(output, 'w:gz') as archive:\n"
        "    for folder in ('artifacts/chronos2_lora_evaluation', 'reports'):\n"
        "        if (PROJECT / folder).exists():\n            archive.add(PROJECT / folder, arcname=folder)\n"
        "print(output)\nif EVALUATION_EXIT:\n    raise RuntimeError('Evaluation failed; inspect exported logs and caches')\n"))
    worker = Path(".cache/kaggle-chronos2-evaluation")
    worker.mkdir(parents=True, exist_ok=True)
    name = "32_chronos2_adaptation_evaluation.ipynb"
    (worker/name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    (worker/"kernel-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-chronos2-adaptation-evaluation",
        "title": "5G NI Chronos2 Adaptation Evaluation", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [submission["input_dataset"], "truongbaokien/5g-ni-chronos2-evaluation-code"],
        "kernel_sources": [submission["kernel"]], "competition_sources": []}, indent=2))
    print(json.dumps({"code_manifest_sha256": digest, "training_manifest_sha256": training_sha,
                      "code_files": len(files), "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
