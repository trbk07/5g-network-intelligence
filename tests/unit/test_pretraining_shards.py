"""Array-only sharded source integrity and sampling checks; no model fitting."""

import json

import numpy as np
import pytest
import torch

from src.data.io import sha256
from src.features.pretraining_batches import PretrainingBatches
from src.features.pretraining_shards import iter_sharded_segments


def fixture_report(tmp_path):
    shards = []
    for index in range(2):
        path = tmp_path/f"shard_{index}.npz"
        values = np.full((200, 1, 4), np.nan)
        values[:, 0, 0] = np.arange(200.)+index
        phase = np.arange(1000+300*index, 1200+300*index)
        name = f"segment_{index}"
        np.savez_compressed(path, **{name: values, name+"_phase": phase})
        shards.append({"processed_path": str(path), "processed_sha256": sha256(path), "segments": [{
            "name": name, "entity": index, "hours": 200, "start_hour": int(phase[0]), "end_hour": int(phase[-1])}]})
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"shards": shards}))
    report = {"format": "sharded_prefix_v1", "processed_path": str(manifest), "processed_sha256": sha256(manifest),
              "training_ready": True, "future_metric_values_read": False}
    report_path = tmp_path/"report.json"
    report_path.write_text(json.dumps(report))
    return report, report_path, shards


def test_shards_remain_one_domain_and_reproducible(tmp_path):
    report, path, _ = fixture_report(tmp_path)
    segments = list(iter_sharded_segments(report))
    assert len(segments) == 2 and {s[0] for s in segments} == {0, 1}
    sampler = PretrainingBatches([path], [1], samples=7)
    assert len(sampler.sources) == 1 and len(sampler.sources[0]) == 2
    first, second = list(sampler.batches(4, 17)), list(sampler.batches(4, 17))
    assert sum(len(x[0]) for x in first) == 7
    for a, b in zip(first, second):
        assert a[0].shape[1:] == (168, 2)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_corrupt_later_shard_fails_before_first_yield(tmp_path):
    report, _, shards = fixture_report(tmp_path)
    from pathlib import Path
    Path(shards[1]["processed_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt shard"):
        next(iter_sharded_segments(report))


def test_unaudited_source_cannot_enter_sampler(tmp_path):
    report, path, _ = fixture_report(tmp_path)
    report["training_ready"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="training-ready"):
        PretrainingBatches([path], [1])


def test_excluded_source_segment_is_accounted_but_never_sampled(tmp_path):
    report, path, shards = fixture_report(tmp_path)
    removed = shards[0]["segments"].pop()
    shards[0]["excluded_segments"] = [{"name": removed["name"], "reason": "constant"}]
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"shards": shards}))
    report["processed_sha256"] = sha256(manifest)
    path.write_text(json.dumps(report))
    segments = list(iter_sharded_segments(report))
    assert len(segments) == 1 and segments[0][0] == 1
    sampler = PretrainingBatches([path], [1], samples=8)
    assert len(sampler.sources) == 1 and len(sampler.sources[0]) == 1
