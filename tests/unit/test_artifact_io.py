"""Artifact integrity/routing failures using statistics-only baseline fixtures; no fitting."""

import json

import pytest

from src.data.io import sha256
from src.inference.registry import load_champion
from tests.utils.demo import baseline_assets


def rehash(root):
    path = root / "artifacts_manifest.json"
    data = json.loads(path.read_text())
    data["files"] = {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p != path}
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("family", ["persistence", "seasonal_naive_24", "seasonal_naive_168"])
def test_baseline_family_routes_without_training(tmp_path, family):
    root = baseline_assets(tmp_path / "demo") / "champion"
    meta_path = root / "champion_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["family"] = family
    meta_path.write_text(json.dumps(meta))
    rehash(root)
    models, thresholds, loaded = load_champion(root)
    assert set(models) == {1, 6, 24}
    assert all(m.name == family for m in models.values())
    assert thresholds.fit_end == loaded["fit_end"] == 199


def test_metadata_runtime_identity_mismatch_refused(tmp_path):
    root = baseline_assets(tmp_path / "demo") / "champion"
    path = root / "champion_meta.json"
    meta = json.loads(path.read_text())
    meta["freeze_commit"] = "different"
    path.write_text(json.dumps(meta))
    rehash(root)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_champion(root)


def test_missing_and_corrupt_files_refused(tmp_path):
    root = baseline_assets(tmp_path / "demo") / "champion"
    (root / "thresholds.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact mismatch"):
        load_champion(root)
    (root / "thresholds.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_champion(root)


def test_gru_cpu_fixture_route_has_phase4_and_no_training(tmp_path):
    """Persist initialized state only; never call fit/backward/optimizer.step."""
    import numpy as np
    import torch

    from src.features.gru_inputs import GRUPreprocessor
    from src.models.gru import GRUNet

    root = baseline_assets(tmp_path / "demo") / "champion"
    meta_path = root / "champion_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(family="gru_A", seeds=[17])
    meta_path.write_text(json.dumps(meta))
    folder = root / "seed17"
    folder.mkdir()
    prep = GRUPreprocessor(np.zeros(1), np.zeros(1), np.ones(1), tuple(meta["beam_ids"]), 199, "A")
    prep.save(folder / "preprocessor.npz")
    architecture = {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}
    model = GRUNet(2, **architecture)
    torch.save({"model": model.state_dict()}, folder / "best.pt")
    (folder / "meta.json").write_text(
        json.dumps({"variant": "A", "seed": 17, "architecture": architecture})
    )
    manifest = json.loads((root / "artifacts_manifest.json").read_text())
    manifest["files"] = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "artifacts_manifest.json"
    }
    (root / "artifacts_manifest.json").write_text(json.dumps(manifest))
    models, _, _ = load_champion(root)
    history = np.ones((168, 1, 4))
    assert models[1].predict_at(history, 199, 1).shape == (1,)
    assert np.isfinite(models[24].predict_at(history, 199, 24)).all()
