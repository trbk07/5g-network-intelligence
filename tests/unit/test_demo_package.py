"""Package install integrity check happens before extraction, without fitting."""

import io
import json
import tarfile

import pytest

from scripts.fetch_demo_assets import install_archive
from src.data.io import sha256


def test_wrong_checksum_does_not_extract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/demo_asset_package.json").write_text(json.dumps({"sha256": "wrong"}))
    path = tmp_path / "demo.tar.gz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_archive(path)
    assert not (tmp_path / "demo_assets").exists()


def test_verified_archive_still_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    path = tmp_path / "demo.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("demo_assets/../../escaped")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    (tmp_path / "reports/demo_asset_package.json").write_text(json.dumps({"sha256": sha256(path)}))
    with pytest.raises(ValueError, match="Unsafe demo"):
        install_archive(path)
    assert not (tmp_path.parent / "escaped").exists()
