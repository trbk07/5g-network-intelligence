"""Archive intake tests with text fixtures only; no fitting or model loading."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.import_hurdle_results import import_results


def archive_fixture(path, extra=None, bad_hash=False):
    model = b"fixture checkpoint bytes, not a trained model"
    files = {
        "artifacts/hurdle_research/inner1/h1/direct.txt": model,
        "artifacts/hurdle_research/manifest.json": json.dumps({"files": {
            "inner1/h1/direct.txt": "bad" if bad_hash else hashlib.sha256(model).hexdigest()
        }}).encode(),
        "reports/hurdle_development_scores.csv": b"fold,horizon,model,mae\n",
        "configs/champion.yaml": b"must never be installed",
        "artifacts/champion/model.txt": b"must never be installed",
    }
    files.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_intake_is_allowlisted_and_idempotent(tmp_path):
    archive = tmp_path / "results.tar.gz"
    archive_fixture(archive)
    first = import_results(archive, tmp_path)
    second = import_results(archive, tmp_path)
    assert first == second
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "artifacts/champion").exists()
    assert first["verified_artifact_files"] == 1


@pytest.mark.parametrize("bad_name", ["../escaped.txt", "artifacts/../../escaped.txt", "C:/escape.txt"])
def test_unsafe_member_rejected_before_install(tmp_path, bad_name):
    archive = tmp_path / "results.tar.gz"
    archive_fixture(archive, {bad_name: b"unsafe"})
    with pytest.raises(ValueError, match="Unsafe"):
        import_results(archive, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_bad_checksum_rejected_before_install(tmp_path):
    archive = tmp_path / "results.tar.gz"
    archive_fixture(archive, bad_hash=True)
    with pytest.raises(ValueError, match="checksum"):
        import_results(archive, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_existing_changed_checkpoint_is_preserved(tmp_path):
    archive = tmp_path / "results.tar.gz"
    archive_fixture(archive)
    target = tmp_path / "artifacts/hurdle_research/inner1/h1/direct.txt"
    target.parent.mkdir(parents=True)
    target.write_text("existing different evidence")
    with pytest.raises(ValueError, match="overwrite"):
        import_results(archive, tmp_path)
    assert target.read_text() == "existing different evidence"
    assert not (tmp_path / "reports/hurdle_development_scores.csv").exists()
