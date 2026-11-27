"""Exercise hosted evaluation packaging with text fixtures, never model code."""

import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import build_chronos2_evaluation_worker as builder


def test_worker_pins_artifacts_and_has_valid_cells(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    for folder in ("src", "scripts", "configs", "reports", "notebooks"):
        (tmp_path/folder).mkdir()
    for name in ("__init__.py", "evaluate_chronos2_lora.py", "train_multisource_neural.py",
                 "train_temporal_mlp.py", "train_temporal_linear.py"):
        (tmp_path/"scripts"/name).write_text("# packaging fixture\n")
    for name in ("data.yaml", "evaluation.yaml", "chronos2_lora.yaml", "chronos2_synth.yaml"):
        (tmp_path/"configs"/name).write_text("fixture: true\n")
    (tmp_path/"LICENSE").write_text("fixture license")
    notebook = repo/"notebooks/32_chronos2_adaptation_evaluation.ipynb"
    (tmp_path/"notebooks"/notebook.name).write_bytes(notebook.read_bytes())
    (tmp_path/"reports/chronos2_lora_cloud_submission.json").write_text(json.dumps({
        "manifest_sha256": "a"*64, "input_dataset": "owner/pinned-input", "kernel": "owner/pinned-fit"}))
    names = []
    for fold in ("inner1", "inner2"):
        names.append(f"{fold}/training_protocol.json")
        for step in (100, 300, 1000):
            names.extend(f"{fold}/trainer/checkpoint-{step}/{name}" for name in
                         ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                          "scheduler.pt", "rng_state.pth", "trainer_state.json"))
    manifest = tmp_path/"training.json"
    manifest.write_text(json.dumps({"checkpoint_selection_pending": True,
                                    "files": dict.fromkeys(names, "b"*64)}))
    monkeypatch.setattr(sys, "argv", ["builder", "--training-manifest", str(manifest)])
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: "fixture-commit\n")
    builder.main()
    worker = tmp_path/".cache/kaggle-chronos2-evaluation"
    generated = json.loads((worker/notebook.name).read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in generated["cells"] if c["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    combined = "\n".join(sources)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() in combined
    assert "RUN_EVALUATION = True" in combined
    assert "EVALUATION_EXIT = process.wait()" in combined
    assert "train_chronos2_lora" not in combined
    metadata = json.loads((worker/"kernel-metadata.json").read_text())
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert metadata["kernel_sources"] == ["owner/pinned-fit"]
    with zipfile.ZipFile(tmp_path/".cache/chronos2-evaluation-source/evaluation_source.zip") as bundle:
        assert not any("pretrained/" in name or "data/" in name for name in bundle.namelist())
    manifest.write_text(json.dumps({"checkpoint_selection_pending": True, "files": {"fake": "bad"}}))
    with pytest.raises(ValueError, match="SHA256"):
        builder.main()
