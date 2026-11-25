"""Package an explicit cloud input bundle with non-development targets redacted."""

import hashlib
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.train_multisource_neural import source_cases
from src.data.panel import load_panel, save_panel


def main():
    root = Path.cwd()
    output = root/".cache/multisource-neural-source"
    output.mkdir(parents=True, exist_ok=True)
    stage = root/".cache/multisource-neural-panel/data/processed"
    panel = load_panel(root/"data/processed/panel.npz")
    # Keep the canonical time axis for frozen-split verification; no future values travel.
    values = np.full(panel.values.shape, np.nan, dtype=panel.values.dtype)
    values[:504] = panel.values[:504]
    save_panel(replace(panel, values=values), stage/"panel.npz")
    files = {p.relative_to(root).as_posix(): p for p in root.glob("src/**/*.py")}
    for name in ("__init__.py", "train_multisource_neural.py", "train_temporal_mlp.py",
                 "train_temporal_linear.py"):
        files[f"scripts/{name}"] = root/"scripts"/name
    for name in ("data.yaml", "evaluation.yaml", "multisource_neural_pilot.yaml"):
        files[f"configs/{name}"] = root/"configs"/name
    for name in ("panel.npz", "panel_meta.json"):
        files[f"data/processed/{name}"] = stage/name
    files["data/processed/split.json"] = root/"data/processed/split.json"
    for name in ("transfer_overlap_audit.json", "multisource_preparation.json",
                 "cesnet_acquisition.json", "cesnet_prefix_audit.json",
                 "multisource_neural_input_check.json"):
        files[f"reports/{name}"] = root/"reports"/name
    reports, _ = source_cases()["radio_and_cesnet"]
    for report_path in reports:
        report = json.loads(Path(report_path).read_text())
        files[report_path] = root/report_path
        processed = report["processed_path"].replace("\\", "/")
        files[processed] = root/processed
    for name in ("LICENSE", "docs/multisource_neural_protocol.md"):
        files[name] = root/name
    files["data/external/zenodo/17815388/record.json"] = root/"data/external/zenodo/17815388/record.json"
    manifest = {"source_commit": subprocess.check_output(
        ["git", "-c", "safe.directory=D:/5G", "rev-parse", "HEAD"], text=True).strip(),
        "original_values_available_through": 503,
        "original_values_504_onward": "NaN_redacted_not_packaged",
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in files.items()}}
    bundle = output/"multisource_neural_source.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("colab_source_manifest.json", json.dumps(manifest, indent=2))
    evidence = {"sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "bytes": bundle.stat().st_size, "files": len(files),
                "source_commit": manifest["source_commit"], "training_performed": False,
                "original_values_available_through": 503}
    (output/"transport.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
