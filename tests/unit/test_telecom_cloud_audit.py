"""Hash and counter fixtures only; no checkpoint training or optimization."""

import json

import pytest
import torch

from scripts.audit_telecom_cloud import verify_steps, verify_tree
from src.data.io import sha256


def test_optimizer_budget_requires_complete_counts():
    cfg = {"pretraining_epochs": 60, "pretraining_samples_per_epoch": 65536, "pretraining_batch_size": 256}
    state = {"epoch": 59, "optimizer": {"state": {0: {"step": torch.tensor(15360)}, 1: {"step": torch.tensor(15360)}}}}
    assert verify_steps(state, cfg) == 15360
    state["optimizer"]["state"][1]["step"] = torch.tensor(15359)
    with pytest.raises(ValueError, match="budget incomplete"):
        verify_steps(state, cfg)
    state["optimizer"]["state"] = {}
    with pytest.raises(ValueError, match="budget incomplete"):
        verify_steps(state, cfg)


def test_manifest_requires_exact_file_coverage(tmp_path):
    payload = tmp_path/"fixture.txt"
    payload.write_text("Not a trained checkpoint")
    (tmp_path/"manifest.json").write_text(json.dumps({"files": {"fixture.txt": sha256(payload)}}))
    verify_tree(tmp_path)
    extra = tmp_path/"extra.txt"
    extra.write_text("unmanifested")
    with pytest.raises(ValueError, match="coverage differs"):
        verify_tree(tmp_path)


def test_manifest_rejects_mutated_payload(tmp_path):
    payload = tmp_path/"fixture.txt"
    payload.write_text("original fixture")
    (tmp_path/"manifest.json").write_text(json.dumps({"files": {"fixture.txt": sha256(payload)}}))
    payload.write_text("changed fixture")
    with pytest.raises(ValueError, match="hash differs"):
        verify_tree(tmp_path)
