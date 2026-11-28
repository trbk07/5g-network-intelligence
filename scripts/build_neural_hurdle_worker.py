"""Package redacted development inputs and a readable private GPU worker."""

import hashlib
import json
import subprocess
import zipfile
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.data.panel import load_panel, save_panel
from src.utils.config import load_config


def main(study="neural_hurdle"):
    if study not in ("neural_hurdle", "mae_aligned", "chronos2_lora", "cell_attention", "cell_multisource", "cell_attention_v2", "cell_multisource_v2"):
        raise ValueError("Unknown study")
    stem = study.replace("_", "-")
    notebook_name = ("28_multivariate_zero_inflated_forecasting.ipynb" if study == "neural_hurdle"
                     else "29_mae_aligned_forecasting.ipynb")
    title = ("5G NI Multivariate Zero Inflated Forecasting" if study == "neural_hurdle"
             else "5G NI MAE Aligned Forecasting")
    slug = ("5g-ni-multivariate-zero-inflated-forecasting" if study == "neural_hurdle"
            else "5g-ni-mae-aligned-forecasting")
    if study == "chronos2_lora":
        notebook_name = "31_chronos2_multivariate_finetuning.ipynb"
        title = "5G NI Chronos2 Multivariate Fine Tuning"
        slug = "5g-ni-chronos2-multivariate-fine-tuning"
    if study == "cell_attention":
        notebook_name = "33_cell_spatial_temporal_attention.ipynb"
        title = "5G NI Cell Spatial Temporal Attention"
        slug = "5g-ni-cell-spatial-temporal-attention"
    if study == "cell_multisource":
        notebook_name = "34_multi_corpus_spatial_encoder_pretraining.ipynb"
        title = "5G NI Multi Corpus Spatial Encoder Pretraining"
        slug = "5g-ni-multi-corpus-spatial-encoder-pretraining"
    if study == "cell_attention_v2":
        notebook_name = "37_seeded_cell_spatial_temporal_attention.ipynb"
        title = "5G NI Seeded Cell Spatial Temporal Attention"
        slug = "5g-ni-seeded-cell-spatial-temporal-attention"
    if study == "cell_multisource_v2":
        notebook_name = "38_reproducible_multi_corpus_encoder_transfer.ipynb"
        title = "5G NI Reproducible Multi Corpus Encoder Transfer"
        slug = "5g-ni-reproducible-multi-corpus-encoder-transfer"
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
    if study in ("cell_attention_v2", "cell_multisource", "cell_multisource_v2"):
        files["scripts/train_cell_attention.py"] = root/"scripts/train_cell_attention.py"
    if study == "cell_multisource_v2":
        files["scripts/train_cell_multisource.py"] = root/"scripts/train_cell_multisource.py"
    if study in ("cell_multisource", "cell_multisource_v2"):
        cfg = load_config(f"configs/{study}.yaml")
        for report_name in cfg["source_reports"]:
            report_path = root/report_name
            report = json.loads(report_path.read_text())
            prepared = Path(report["processed_path"])
            files[report_name] = report_path
            files[prepared.as_posix()] = root/prepared
        for name in ("transfer_overlap_audit.json", "multisource_preparation.json"):
            files["reports/"+name] = root/"reports"/name
    if study == "chronos2_lora":
        model_cfg = load_config("configs/chronos2_synth.yaml")
        model_root = root/"data/external/pretrained/chronos2_synth"/model_cfg["revision"]
        license_path = model_root/"LICENSE.apache-2.0.txt"
        if not license_path.exists():
            with urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=60) as response:
                license_text = response.read()
            if b"Version 2.0, January 2004" not in license_text:
                raise ValueError("Expected Apache license text")
            license_path.write_bytes(license_text)
        for name in ("config.json", "README.md", "model.safetensors", "LICENSE.apache-2.0.txt"):
            files[f"pretrained/chronos2_synth/{name}"] = model_root/name
        files["configs/chronos2_synth.yaml"] = root/"configs/chronos2_synth.yaml"
        files["docs/foundation_finetuning_research.md"] = root/"docs/foundation_finetuning_research.md"
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "original_values_available_through": 503,
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    bundle = source/f"{study}_source.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if name.endswith(".safetensors") else zipfile.ZIP_DEFLATED)
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
    if study in ("cell_multisource", "cell_multisource_v2"):
        (source/"DATA_SOURCES.md").write_text(
            "# Sources and licenses\n\nCode and NETOP GOTS-F: MIT. "
            "NETOP: https://huggingface.co/datasets/netop/gotsf-ds.\n\n"
            "External PM: Peter Lehocky et al., https://zenodo.org/records/17815388, CC BY 4.0.\n"
            "CESNET-TimeSeries24: Koumar, Hynek, Cejka, Siska; https://zenodo.org/records/13382427, CC BY 4.0.\n"
            "Vigo: Fondo-Ferreiro, Rodelgo-Lacruz, Gonzalez-Castano, Gil-Castineira, Candal-Ventureira; "
            "https://zenodo.org/records/8147768, CC BY 4.0.\n"
            "CESNET-AGG23: Benes, Pesek, Cejka; https://zenodo.org/records/8053021, CC BY 4.0.\n"
            "Vigo/AGG23 HDF5 conversion: Koumar and Jerabek, https://zenodo.org/records/22721975.\n"
            "Changes: hourly aggregation where needed, chronological prefix filtering, gap-preserving segment export. "
            "External data retains CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. "
            "Traffic values are representation-pretraining inputs, not original PRB labels.\n")
    if study == "chronos2_lora":
        with (source/"DATA_SOURCES.md").open("a") as handle:
            handle.write("\nPretrained weights: autogluon/chronos-2-synth, Apache-2.0. "
                "Pinned publisher README and full Apache license are included under pretrained/chronos2_synth. "
                "Project code and GOTS-F remain under their MIT licenses.\n")
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id": f"truongbaokien/5g-ni-{stem}-inputs", "title": f"5G NI {stem.title()} Inputs",
        "licenses": [{"name": "Apache 2.0" if study == "chronos2_lora" else "CC-BY-4.0" if study in ("cell_multisource", "cell_multisource_v2") else "MIT"}]}, indent=2))
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
        "    if name.startswith('pretrained/'):\n        continue  # Keep immutable base weights in Input.\n"
        "    target = project / name\n    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(input_root / name, target)\n"
        "os.chdir(project)\nsys.path.insert(0, str(project))\n"
        "os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']\n"
        "print('Verified source:', manifest['source_commit'])\n")
    if study == "chronos2_lora":
        setup += "MODEL_ROOT = input_root / 'pretrained/chronos2_synth'\n"
    notebook["cells"].insert(4, {"cell_type": "code", "id": "verified-worker-inputs",
        "metadata": {}, "source": setup.splitlines(keepends=True), "execution_count": None, "outputs": []})
    if study == "chronos2_lora":
        install = ("# Install pinned official libraries before importing model code.\n"
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',\n"
            "                'chronos-forecasting==2.3.2', 'transformers==4.57.6', 'peft==0.18.1'], check=True)\n")
        notebook["cells"].insert(5, {"cell_type": "code", "id": "pinned-finetuning-environment",
            "metadata": {}, "source": install.splitlines(keepends=True), "execution_count": None, "outputs": []})
    export = ("# Preserve completed outputs for independent CPU replay.\nimport tarfile\n"
        f"output = Path('/kaggle/working/{study}_results.tar.gz')\n"
        "with tarfile.open(output, 'w:gz') as archive:\n"
        f"    for folder in ('artifacts/{study}', 'reports'):\n"
        "        if (PROJECT / folder).exists():\n            archive.add(PROJECT / folder, arcname=folder)\nprint(output)\n")
    if study in ("chronos2_lora", "cell_attention", "cell_multisource", "cell_attention_v2", "cell_multisource_v2"):
        export += "if TRAIN_EXIT:\n    raise RuntimeError('Cloud fine-tuning failed; inspect saved logs/checkpoints before restarting')\n"
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
        "enable_gpu": True, "enable_internet": study == "chronos2_lora",
        "dataset_sources": [f"truongbaokien/5g-ni-{stem}-inputs"],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    print(json.dumps(transport, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["neural_hurdle", "mae_aligned", "chronos2_lora", "cell_attention", "cell_multisource", "cell_attention_v2", "cell_multisource_v2"], default="neural_hurdle")
    main(parser.parse_args().study)
