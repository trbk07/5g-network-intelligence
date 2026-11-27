"""Pure decision gates on fixed numerical fixtures; no coefficient selection."""

import pytest

from scripts.select_clipped_blend import strict_acceptance


def rows():
    return [{"horizon": h, "mae": .999, "current_mae": 1., "clipped_weight": .1} for h in (1, 6, 24)]


def test_strict_gate_rejects_any_regression_or_zero_only_choices():
    data = rows()
    assert strict_acceptance(data, .0009)
    data[2]["mae"] = 1.001
    assert not strict_acceptance(data, .0009)
    data = rows()
    for row in data:
        row["clipped_weight"] = 0
    assert not strict_acceptance(data, .0009)


def test_gate_requires_complete_horizons_and_material_gain():
    assert not strict_acceptance(rows(), .01)
    with pytest.raises(ValueError, match="three"):
        strict_acceptance(rows()[:2], .001)


def test_selection_entrypoint_refuses_local(monkeypatch):
    from scripts import select_clipped_blend

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(select_clipped_blend, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        select_clipped_blend.main()


def test_intake_requires_current_control_report_before_copy(tmp_path):
    import hashlib
    import io
    import json
    import tarfile

    from scripts.import_hurdle_results import import_results

    archive = tmp_path/"result.tar.gz"
    content = b"fixed selection fixture"
    files = {"artifacts/clipped_blend/selection.json": content,
             "artifacts/clipped_blend/manifest.json": json.dumps({"files": {
                 "selection.json": hashlib.sha256(content).hexdigest()}}).encode(),
             "reports/clipped_blend_selection_grid.csv": b"grid fixture",
             "reports/clipped_blend_confirmation.csv": b"confirmation fixture"}

    def write():
        with tarfile.open(archive, "w:gz") as output:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))

    write()
    with pytest.raises(ValueError, match="Complete score"):
        import_results(archive, tmp_path, "clipped_blend")
    assert not (tmp_path/"artifacts").exists()
    files["reports/clipped_blend_current_control.csv"] = b"reference fixture"
    write()
    assert import_results(archive, tmp_path, "clipped_blend")["copied_or_identical_files"] == 5
