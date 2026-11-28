"""Adapt baseline worker to existing approved hosted inputs without re-exporting data."""

import hashlib
import json
import pprint
import zipfile
from pathlib import Path

import yaml


def main():
    baseline = Path(".cache/cell-attention-v2-source/cell_attention_v2_source.zip")
    shared = Path(".cache/cell-multisource-v2-source/cell_multisource_v2_source.zip")
    with zipfile.ZipFile(baseline) as archive:
        first = json.loads(archive.read("source_manifest.json"))
    with zipfile.ZipFile(shared) as archive:
        second = json.loads(archive.read("source_manifest.json"))
    panel_name = "data/processed/panel.npz"
    if first["files"][panel_name] != second["files"][panel_name]:
        raise ValueError("Hosted payload is not identical to the proposed baseline data")
    transport = json.loads(Path(".cache/cell-multisource-v2-source/transport.json").read_text())
    previous = json.loads(Path(".cache/cell-attention-v2-source/transport.json").read_text())
    stage = Path(".cache/kaggle-cell-attention-v2")
    path = stage/"37_seeded_cell_spatial_temporal_attention.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    config_path = Path("configs/cell_attention_v2.yaml")
    text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    expected = hashlib.sha256(text.encode()).hexdigest()
    if yaml.safe_dump(cfg, sort_keys=False) != text:
        raise ValueError("Configuration round-trip differs")
    command = ("from scripts.train_cell_attention import run_study\n"
               "run_study('configs/cell_attention_v2.yaml', 'artifacts/cell_attention_v2', 'cell_attention_v2')")
    original_call = "[sys.executable, '-u', '-m', 'scripts.train_cell_attention_v2']"
    replacement = "[sys.executable, '-u', '-c', "+repr(command)+"]"
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        source = source.replace(previous["manifest_sha256"], transport["manifest_sha256"])
        if original_call in source:
            source = source.replace(original_call, replacement)
            changed = True
        cell["source"] = source.splitlines(keepends=True)
    if not changed:
        raise ValueError("Expected baseline invocation not found; do not rewrite unknown worker")
    setup = ("# Reuse already hosted inputs; materialize only the declared public configuration.\n"
        "import yaml, hashlib\nconfiguration = "+pprint.pformat(cfg, sort_dicts=False, width=92)+"\n"
        "config_path = project/'configs/cell_attention_v2.yaml'\n"
        "config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding='utf-8')\n"
        f"assert hashlib.sha256(config_path.read_bytes()).hexdigest() == {expected!r}\n"
        "print('Verified baseline configuration; no additional data upload.')\n")
    notebook["cells"].insert(5, {"cell_type": "code", "id": "reuse-hosted-inputs", "metadata": {},
        "source": setup.splitlines(keepends=True), "execution_count": None, "outputs": []})
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = stage/"kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_sources"] = ["truongbaokien/5g-ni-cell-multisource-v2-inputs"]
    metadata_path.write_text(json.dumps(metadata, indent=2))
    evidence = {"reuses_existing_hosted_dataset": metadata["dataset_sources"][0],
                "hosted_manifest_sha256": transport["manifest_sha256"],
                "identical_panel_sha256": second["files"][panel_name],
                "baseline_config_sha256": expected, "new_data_export_performed": False}
    Path("reports/seeded_baseline_hosted_input_reuse.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
