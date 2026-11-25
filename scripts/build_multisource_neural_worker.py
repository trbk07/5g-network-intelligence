"""Stage a readable private Kaggle worker from the separate review notebook."""

import json
import hashlib
import shutil
import zipfile
from pathlib import Path


def main():
    source = Path(".cache/multisource-neural-source")
    with zipfile.ZipFile(source/"multisource_neural_source.zip") as archive:
        manifest_sha = hashlib.sha256(archive.read("colab_source_manifest.json")).hexdigest()
    shutil.copyfile("scripts/run_colab_tcn.py", source/"run_colab_tcn.py")
    (source/"DATA_SOURCES.md").write_text(
        "# Source attribution\n\nCode: project MIT LICENSE in the bundle.\n"
        "Target: NETOP GOTS-F, https://github.com/netop/gotsf-ds (MIT).\n"
        "Radio PM: Peter Lehocky et al., https://zenodo.org/records/17815388, CC BY 4.0; "
        "full publisher attribution is preserved in data/external/zenodo/17815388/record.json.\n"
        "CESNET-TimeSeries24: Josef Koumar, Karel Hynek, Tomas Cejka, Pavel Siska, "
        "https://zenodo.org/records/13382427; hourly distribution "
        "https://zenodo.org/records/23296272, CC BY 4.0. Creator metadata is preserved "
        "in reports/cesnet_acquisition.json.\n\n"
        "Prepared external data retain only chronological training prefixes. "
        "Derived arrays preserve gaps and timestamps; no interpolation. Target "
        "values from hour 504 onward are replaced by NaN. No source is endorsed "
        "by its inclusion in this research pilot.\n", encoding="utf-8")
    (source/"dataset-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-multisource-neural-inputs",
        "title": "5G NI Multisource Neural Inputs", "licenses": [{"name": "CC-BY-4.0"}]}, indent=2))
    notebook = json.loads(Path("notebooks/27_multisource_representation_pretraining.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "RUN_TRAINING = False" in text:
            cell["source"] = text.replace("RUN_TRAINING = False", "RUN_TRAINING = True").splitlines(keepends=True)
    setup = (
        "# Verify the attached input bundle before extracting readable project files.\n"
        "from pathlib import Path, PurePosixPath\nimport os, sys, json, hashlib, shutil\n"
        "# Kaggle extracts uploaded ZIPs into a directory during dataset ingestion.\n"
        "candidates = list(Path('/kaggle/input').rglob('colab_source_manifest.json'))\n"
        "if len(candidates) != 1:\n    raise RuntimeError('Exactly one declared source manifest required')\n"
        "manifest_path = candidates[0]\n"
        f"if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != {manifest_sha!r}:\n"
        "    raise ValueError('Pinned source manifest checksum differs')\n"
        "manifest = json.loads(manifest_path.read_text())\ninput_root = manifest_path.parent\n"
        "project = Path('/kaggle/working/multisource_neural')\n"
        "project.mkdir(exist_ok=True)\n"
        "for name, digest in manifest['files'].items():\n"
        "    relative = PurePosixPath(name)\n"
        "    if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\\\' in name:\n"
        "        raise ValueError('Unsafe manifest path')\n"
        "    original = input_root / name\n"
        "    if not original.resolve().is_relative_to(input_root.resolve()):\n"
        "        raise ValueError('Input path escapes verified root')\n"
        "    if hashlib.sha256(original.read_bytes()).hexdigest() != digest:\n"
        "        raise ValueError('Source checksum differs: ' + name)\n"
        "# Copy only verified members; never import code before verification.\n"
        "for name in manifest['files']:\n"
        "    destination = project / name\n    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(input_root / name, destination)\n"
        "os.chdir(project)\nsys.path.insert(0, str(project))\n"
        "os.environ['SOURCE_GIT_COMMIT'] = manifest['source_commit']\n"
        "print('Verified source commit:', manifest['source_commit'])\n"
        "print('Original values available through:', manifest['original_values_available_through'])\n")
    notebook["cells"].insert(4, {"cell_type": "code", "id": "verified-cloud-inputs",
        "metadata": {}, "source": setup.splitlines(keepends=True), "outputs": [], "execution_count": None})
    export = ("# Retain checkpoints and reports in Kaggle Output for independent verification.\n"
        "import tarfile\noutput = Path('/kaggle/working/multisource_neural_results.tar.gz')\n"
        "with tarfile.open(output, 'w:gz') as archive:\n"
        "    for folder in ('artifacts/multisource_neural', 'reports'):\n"
        "        archive.add(PROJECT / folder, arcname=folder)\nprint(output)\n")
    notebook["cells"].append({"cell_type": "code", "id": "durable-output-export", "metadata": {},
                              "source": export.splitlines(keepends=True), "outputs": [], "execution_count": None})
    stage = Path(".cache/kaggle-multisource-neural")
    stage.mkdir(parents=True, exist_ok=True)
    name = "27_multisource_representation_pretraining.ipynb"
    (stage/name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage/"kernel-metadata.json").write_text(json.dumps({
        "id": "truongbaokien/5g-ni-multisource-representation-pretraining",
        "title": "5G NI Multisource Representation Pretraining", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["truongbaokien/5g-ni-multisource-neural-inputs"],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    print(stage)


if __name__ == "__main__":
    main()
