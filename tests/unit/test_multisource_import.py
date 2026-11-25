"""Static archive fixtures exercise complete multisource intake without fitting."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.import_hurdle_results import import_results


def archive_fixture(path, missing=None):
    cases = ["target_only", "target_self_pretrain", "radio", "cesnet", "radio_and_cesnet"]
    model = b"static checkpoint fixture; no trained weights"
    files = {"artifacts/multisource_neural/target_only/inner1/A/best.pt": model,
             "artifacts/multisource_neural/manifest.json": json.dumps({"files": {
                 "target_only/inner1/A/best.pt": hashlib.sha256(model).hexdigest()}}).encode(),
             "artifacts/champion/forbidden.txt": b"must not be imported",
             "reports/cesnet_preparation.json": b"must not overwrite provenance"}
    files.update({f"reports/multisource_neural_{case}_development.csv": b"fixture table\n"
                  for case in cases if case != missing})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_complete_multisource_intake_stays_allowlisted(tmp_path):
    archive = tmp_path/"result.tar.gz"
    archive_fixture(archive)
    result = import_results(archive, tmp_path, "multisource_neural")
    assert result["verified_artifact_files"] == 1
    assert result["copied_or_identical_files"] == 7
    assert not (tmp_path/"artifacts/champion").exists()
    assert not (tmp_path/"reports/cesnet_preparation.json").exists()
    assert import_results(archive, tmp_path, "multisource_neural") == result


def test_missing_control_rejects_before_copy(tmp_path):
    archive = tmp_path/"result.tar.gz"
    archive_fixture(archive, missing="target_self_pretrain")
    with pytest.raises(ValueError, match="Complete score"):
        import_results(archive, tmp_path, "multisource_neural")
    assert not (tmp_path/"artifacts").exists()
