import json

import pytest

from src.data.io import build_manifest, sha256
from src.utils.config import load_config


def test_manifest_counts_and_tamper_detection(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("Time,0_0_0\n0,1\n1,2\n")
    (tmp_path / "provenance.json").write_text(json.dumps({
        "revision": "fixed", "license": "mit", "checksums": {"sample.csv": sha256(path)}}))
    manifest = build_manifest(tmp_path)
    assert manifest["files"][0]["rows"] == 2
    assert manifest["files"][0]["columns"] == ["Time", "0_0_0"]
    path.write_text("Time,0_0_0\n0,9\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        build_manifest(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing="):
        build_manifest(tmp_path)


def test_config_rejects_list(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ValueError):
        load_config(path)
