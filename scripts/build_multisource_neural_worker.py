"""Stage a readable private Kaggle worker from the separate review notebook."""

import json
import shutil
from pathlib import Path


def main():
    source = Path(".cache/multisource-neural-source")
    transport = json.loads((source/"transport.json").read_text())
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
        "from pathlib import Path\nimport os, sys, zipfile\n"
        "candidates = list(Path('/kaggle/input').rglob('multisource_neural_source.zip'))\n"
        "if len(candidates) != 1:\n    raise RuntimeError('Exactly one declared source bundle required')\n"
        "bundle = candidates[0]\nsys.path.insert(0, str(bundle.parent))\n"
        "from run_colab_tcn import verify_transport\n"
        f"manifest = verify_transport(bundle, {transport['sha256']!r})\n"
        "project = Path('/kaggle/working/multisource_neural')\n"
        "project.mkdir(exist_ok=True)\n"
        "with zipfile.ZipFile(bundle) as archive:\n    archive.extractall(project)\n"
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
        "id": "truongbaokien/5g-ni-multisource-neural-pretraining",
        "title": "5G NI Multisource Representation Pretraining", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["truongbaokien/5g-ni-multisource-neural-inputs"],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    print(stage)


if __name__ == "__main__":
    main()
