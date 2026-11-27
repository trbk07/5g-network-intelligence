"""Static archive fixtures exercise complete multisource intake without fitting."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.import_hurdle_results import import_results


def archive_fixture(path, missing=None, study="multisource_neural"):
    cases = {"neural_hurdle": ["direct", "hurdle"],
             "mae_aligned": ["direct_clipped", "hurdle_mae"],
             "multisource_neural": ["target_only", "target_self_pretrain", "radio", "cesnet", "radio_and_cesnet"]}[study]
    model = b"static checkpoint fixture; no trained weights"
    files = {f"artifacts/{study}/{cases[0]}/inner1/A/best.pt": model,
             f"artifacts/{study}/manifest.json": json.dumps({"files": {
                 f"{cases[0]}/inner1/A/best.pt": hashlib.sha256(model).hexdigest()}}).encode(),
             "artifacts/champion/forbidden.txt": b"must not be imported",
             "reports/cesnet_preparation.json": b"must not overwrite provenance"}
    files.update({f"reports/{study}_{case}_development.csv": b"fixture table\n"
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


def test_neural_hurdle_requires_both_heads(tmp_path):
    archive = tmp_path/"result.tar.gz"
    archive_fixture(archive, missing="hurdle", study="neural_hurdle")
    with pytest.raises(ValueError, match="Complete score"):
        import_results(archive, tmp_path, "neural_hurdle")
    archive_fixture(archive, study="neural_hurdle")
    result = import_results(archive, tmp_path, "neural_hurdle")
    assert result["copied_or_identical_files"] == 4
    assert not (tmp_path/"artifacts/champion").exists()


def test_mae_aligned_requires_both_objectives(tmp_path):
    archive = tmp_path/"result.tar.gz"
    archive_fixture(archive, missing="hurdle_mae", study="mae_aligned")
    with pytest.raises(ValueError, match="Complete score"):
        import_results(archive, tmp_path, "mae_aligned")
    archive_fixture(archive, study="mae_aligned")
    assert import_results(archive, tmp_path, "mae_aligned")["copied_or_identical_files"] == 4
    assert not (tmp_path/"artifacts/champion").exists()
