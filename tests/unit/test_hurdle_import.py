"""Archive intake tests with text fixtures only; no fitting or model loading."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.import_hurdle_results import import_results


def archive_fixture(path, extra=None, bad_hash=False, candidate=False, calibration=False, temporal_linear=False, blend=False, blend_candidate=False, chronos=False):
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
    if candidate:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/hurdle_candidate/")
                 .replace("reports/hurdle_development_scores.csv", "reports/hurdle_candidate_validation.csv"): value
                 for name, value in files.items()}
        files["reports/forecasts/hurdle_candidate_val.npz"] = b"test forecast bytes, not a model"
    if calibration:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/hurdle_calibration/")
                 .replace("reports/hurdle_development_scores.csv", "reports/hurdle_calibration_development.csv"): value
                 for name, value in files.items()}
    if temporal_linear:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/temporal_linear/")
                 .replace("reports/hurdle_development_scores.csv", "reports/temporal_linear_development.csv"): value
                 for name, value in files.items()}
    if blend:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/development_blend/")
                 .replace("reports/hurdle_development_scores.csv", "reports/development_blend_selection_grid.csv"): value
                 for name, value in files.items()}
        files["reports/development_blend_confirmation.csv"] = b"test confirmation table\n"
    if blend_candidate:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/blend_candidate/")
                 .replace("reports/hurdle_development_scores.csv", "reports/blend_candidate_validation.csv"): value
                 for name, value in files.items()}
        files["reports/forecasts/blend_candidate_val.npz"] = b"test forecast fixture\n"
    if chronos:
        files = {name.replace("artifacts/hurdle_research/", "artifacts/chronos_zero_shot/")
                 .replace("reports/hurdle_development_scores.csv", "reports/chronos_zero_shot_development.csv"): value
                 for name,value in files.items()}
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


def test_mlp_blend_import_requires_both_reports(tmp_path):
    archive = tmp_path / "mlp-blend.tar.gz"
    archive_fixture(archive, blend=True)
    with tarfile.open(archive,"r:gz") as source:
        entries=[(m.name.replace("development_blend","mlp_blend"),source.extractfile(m).read()) for m in source.getmembers()]
    with tarfile.open(archive,"w:gz") as output:
        for name,content in entries:
            member=tarfile.TarInfo(name)
            member.size=len(content)
            output.addfile(member,io.BytesIO(content))
    result=import_results(archive,tmp_path,"mlp_blend")
    assert result["verified_artifact_files"]==1
    assert (tmp_path/"reports/mlp_blend_confirmation.csv").is_file()
    assert (tmp_path/"reports/mlp_blend_selection_grid.csv").is_file()
    assert not (tmp_path/"artifacts/champion").exists()


@pytest.mark.parametrize("kind,prefix", [("temporal_mlp", "temporal_mlp"), ("quantile_resolution", "hurdle_quantile_resolution")])
def test_new_study_uses_its_own_allowlist(tmp_path, kind, prefix):
    archive = tmp_path / "mlp.tar.gz"
    archive_fixture(archive, temporal_linear=True)
    # Rewrite fixture paths only; these bytes never contain a fitted model.
    with tarfile.open(archive, "r:gz") as source:
        entries = [(m.name.replace("temporal_linear", prefix), source.extractfile(m).read())
                   for m in source.getmembers()]
    with tarfile.open(archive, "w:gz") as output:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    evidence = import_results(archive, tmp_path, kind)
    assert evidence["verified_artifact_files"] == 1
    assert (tmp_path / f"reports/{prefix}_import_evidence.json").is_file()
    assert not (tmp_path / "artifacts/champion").exists()


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


def test_candidate_import_requires_separate_allowlist_and_evidence(tmp_path):
    archive = tmp_path / "candidate.tar.gz"
    archive_fixture(archive, candidate=True)
    evidence = import_results(archive, tmp_path, kind="candidate")
    assert evidence["verified_artifact_files"] == 1
    assert (tmp_path / "reports/forecasts/hurdle_candidate_val.npz").exists()
    assert (tmp_path / "reports/hurdle_candidate_import_evidence.json").exists()
    assert not (tmp_path / "artifacts/hurdle_research").exists()
    assert not (tmp_path / "artifacts/champion").exists()


def test_calibration_intake_keeps_development_models_separate(tmp_path):
    archive = tmp_path / "calibration.tar.gz"
    archive_fixture(archive, calibration=True)
    import_results(archive, tmp_path, kind="calibration")
    assert (tmp_path / "reports/hurdle_calibration_import_evidence.json").exists()
    assert (tmp_path / "reports/hurdle_calibration_development.csv").exists()
    assert not (tmp_path / "artifacts/hurdle_research").exists()
    assert not (tmp_path / "artifacts/champion").exists()


def test_temporal_intake_does_not_touch_hurdle_or_champion(tmp_path):
    archive = tmp_path / "temporal.tar.gz"
    archive_fixture(archive, temporal_linear=True)
    import_results(archive, tmp_path, kind="temporal_linear")
    assert (tmp_path / "reports/temporal_linear_development.csv").exists()
    assert (tmp_path / "reports/temporal_linear_import_evidence.json").exists()
    assert not (tmp_path / "artifacts/hurdle_research").exists()
    assert not (tmp_path / "artifacts/champion").exists()


def test_blend_intake_keeps_base_model_roots_untouched(tmp_path):
    archive = tmp_path / "blend.tar.gz"
    archive_fixture(archive, blend=True)
    import_results(archive, tmp_path, kind="development_blend")
    assert (tmp_path / "reports/development_blend_confirmation.csv").exists()
    assert (tmp_path / "reports/development_blend_import_evidence.json").exists()
    assert not (tmp_path / "artifacts/hurdle_research").exists()
    assert not (tmp_path / "artifacts/champion").exists()


def test_fixed_blend_intake_preserves_original_source_roots(tmp_path):
    archive = tmp_path / "fixed-blend.tar.gz"
    archive_fixture(archive, blend_candidate=True)
    import_results(archive, tmp_path, kind="blend_candidate")
    assert (tmp_path / "reports/forecasts/blend_candidate_val.npz").exists()
    assert (tmp_path / "reports/blend_candidate_import_evidence.json").exists()
    assert not (tmp_path / "artifacts/hurdle_research").exists()
    assert not (tmp_path / "artifacts/champion").exists()


def test_zero_shot_intake_never_installs_model_weights(tmp_path):
    archive=tmp_path/"zero-shot.tar.gz"
    archive_fixture(archive,chronos=True,extra={"pretrained_checkpoint/model.safetensors":b"excluded checkpoint bytes"})
    import_results(archive,tmp_path,kind="chronos_zero_shot")
    assert (tmp_path/"reports/chronos_zero_shot_development.csv").exists()
    assert (tmp_path/"reports/chronos_zero_shot_import_evidence.json").exists()
    assert not (tmp_path/"pretrained_checkpoint").exists()
    assert not (tmp_path/"artifacts/champion").exists()
