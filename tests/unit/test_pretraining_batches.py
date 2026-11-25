"""Pure window/provenance tests; no fitting or backward operations."""

import json

import numpy as np
import pytest
import torch

from src.data.io import sha256
from src.features.pretraining_batches import PretrainingBatches


def source(tmp_path, gap=False):
    path = tmp_path / "source.npz"
    values = np.full((220, 1, 4), np.nan)
    values[:, 0, 0] = np.arange(220)
    hours = np.arange(220) + 10000
    if gap:
        hours[100:] += 1
    np.savez(path, segment_0=values, segment_0_phase=hours)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"processed_path": str(path), "processed_sha256": sha256(path),
                                 "segments": [{"name": "segment_0", "entity": "one"}]}))
    return report


def test_causal_windows_targets_and_determinism(tmp_path):
    batches = PretrainingBatches([source(tmp_path)], [1], samples=9)
    first = list(batches.batches(4, 17))
    second = list(batches.batches(4, 17))
    assert [len(x) for x, _, _ in first] == [4, 4, 1]
    for (x, phase, y), repeat in zip(first, second):
        assert x.shape[1:] == (168, 2)
        assert phase.shape[1:] == (4,)
        assert torch.all(x[:, :, 1] == 0)
        scale = batches.provenance[0]["scale"]
        torch.testing.assert_close((y-x[:, -1, :1])*scale,
                                   torch.tensor([[1., 6., 24.]]).expand(len(x), -1),
                                   atol=2e-5, rtol=1e-5)
        for a, b in zip((x, phase, y), repeat):
            assert torch.equal(a, b)


def test_gap_rejected(tmp_path):
    with pytest.raises(ValueError, match="continuous"):
        PretrainingBatches([source(tmp_path, gap=True)], [1])


def test_checksum_rejected(tmp_path):
    report = source(tmp_path)
    data = json.loads(report.read_text())
    data["processed_sha256"] = "0"*64
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        PretrainingBatches([report], [1])


def test_training_entrypoint_rejects_local_before_loading_data(monkeypatch):
    from scripts import train_multisource_neural

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(train_multisource_neural, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        train_multisource_neural.main()


def test_collection_balance():
    from scripts.train_multisource_neural import source_cases

    reports, weights = source_cases()["radio_and_cesnet"]
    assert len(reports) == 4
    assert sum(weights[:3]) == weights[3]
