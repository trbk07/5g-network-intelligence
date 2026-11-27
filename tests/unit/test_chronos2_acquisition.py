"""Tiny mocked publisher payloads test integrity; no real weights or models."""

import hashlib
import io
import json

import pytest

from scripts import acquire_chronos2_synth as acquisition


def setup(monkeypatch, tmp_path, license="apache-2.0", corrupt=False):
    weights, configuration = b"static weight fixture", b'{"model_type":"t5"}'
    cfg = {"model_id": "fixture/synth", "revision": "a"*40, "license": "apache-2.0",
           "weights_file": "model.safetensors", "weights_sha256": hashlib.sha256(weights).hexdigest(),
           "weights_bytes": len(weights), "config_sha256": hashlib.sha256(configuration).hexdigest()}
    metadata = {"sha": cfg["revision"], "cardData": {"license": license}, "siblings": [
        {"rfilename": "model.safetensors", "size": len(weights), "lfs": {"sha256": cfg["weights_sha256"]}}]}

    def open_url(url, **kwargs):
        if "/api/models/" in url:
            return io.BytesIO(json.dumps(metadata).encode())
        return io.BytesIO({"README.md": b"fixture model card", "config.json": configuration,
                           "model.safetensors": b"X"+weights[1:] if corrupt else weights}[url.rsplit("/", 1)[-1]])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acquisition, "load_config", lambda path: cfg)
    monkeypatch.setattr(acquisition.urllib.request, "urlopen", open_url)
    return tmp_path/"data/external/pretrained/chronos2_synth"/cfg["revision"]


def test_verified_fixture_acquisition_never_loads_model(tmp_path, monkeypatch):
    directory = setup(monkeypatch, tmp_path)
    acquisition.main()
    assert (directory/"model.safetensors").exists()
    report = json.loads((tmp_path/"reports/chronos2_synth_acquisition.json").read_text())
    assert not report["training_performed"] and not report["model_instantiated_locally"]


def test_corrupt_payload_is_not_promoted(tmp_path, monkeypatch):
    directory = setup(monkeypatch, tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="safetensors checksum"):
        acquisition.main()
    assert not (directory/"model.safetensors").exists()
    assert (directory/"model.safetensors.part").exists()


def test_license_must_match_before_acquiring_files(tmp_path, monkeypatch):
    directory = setup(monkeypatch, tmp_path, license="unverified")
    with pytest.raises(ValueError, match="license"):
        acquisition.main()
    assert not directory.exists()
