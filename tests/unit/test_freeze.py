"""Pre-holdout guards must fail before any final observations are read."""

import json

import pytest
import yaml

from src.inference.freeze import (
    claim_holdout,
    config_hash,
    source_hash,
    verify_artifacts,
    verify_frozen,
)


def test_freeze_rejects_source_config_and_commit_tampering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "frozen-commit")
    config = {"frozen": True, "holdout_allowed": True}
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    path = tmp_path / "champion.yaml"
    path.write_text(yaml.safe_dump(config))
    manifest = {
        "tag": "pre-holdout",
        "git_tag_verified": True,
        "commit": "frozen-commit",
        "config_sha256": config_hash(config),
        "protected_sources": {"source.py": source_hash(source)},
    }
    attestation = tmp_path / "freeze.json"
    attestation.write_text(json.dumps(manifest))
    verify_frozen(str(path), str(attestation))
    source.write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="source mismatch"):
        verify_frozen(str(path), str(attestation))
    source.write_text("x = 1\n")
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "another-commit")
    with pytest.raises(RuntimeError, match="Executing commit"):
        verify_frozen(str(path), str(attestation))
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "frozen-commit")
    path.write_text(yaml.safe_dump({**config, "changed": True}))
    with pytest.raises(RuntimeError, match="config hash"):
        verify_frozen(str(path), str(attestation))


def test_duplicate_attempt_and_corrupt_artifact(tmp_path):
    marker = tmp_path / "attempt.json"
    claim_holdout(marker, "abc")
    with pytest.raises(RuntimeError, match="already exists"):
        claim_holdout(marker, "abc")
    (tmp_path / "artifacts_manifest.json").write_text(
        json.dumps({"files": {"model.txt": "incorrect"}})
    )
    (tmp_path / "model.txt").write_text("corrupt")
    with pytest.raises(RuntimeError, match="artifact mismatch"):
        verify_artifacts(tmp_path)


def test_holdout_guard_precedes_panel_access(monkeypatch):
    from scripts import run_holdout

    def blocked(*args):
        raise RuntimeError("freeze refused")

    def forbidden(*args, **kwargs):
        pytest.fail("Read final panel before verifying freeze")

    monkeypatch.setattr(run_holdout, "verify_frozen", blocked)
    monkeypatch.setattr(run_holdout, "load_panel", forbidden)
    monkeypatch.setattr("sys.argv", ["run_holdout"])
    with pytest.raises(RuntimeError, match="freeze refused"):
        run_holdout.main()
