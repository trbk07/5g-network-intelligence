"""Read audited external segments without loading four-channel shards together."""

import json
from pathlib import Path

import numpy as np

from src.data.io import sha256


def iter_sharded_segments(report):
    """Yield entity/value/hour triples from a verified, training-ready manifest.

    A source remains one sampling domain regardless of its number of shards.
    Every shard checksum is checked before yielding the first segment.
    """
    if report.get("training_ready") is not True or report.get("future_metric_values_read") is not False:
        raise ValueError("Audited training-ready prefix required")
    path = Path(report["processed_path"])
    if sha256(path) != report["processed_sha256"]:
        raise ValueError("Shard manifest checksum mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shards = manifest["shards"]
    if not shards:
        raise ValueError("No prepared shards")
    seen_paths, seen_names = set(), set()
    for shard in shards:
        payload = Path(shard["processed_path"])
        if (not payload.resolve().is_relative_to(path.parent.resolve()) or payload.is_symlink()
                or payload.resolve() in seen_paths or sha256(payload) != shard["processed_sha256"]):
            raise ValueError("Unsafe, duplicate or corrupt shard")
        seen_paths.add(payload.resolve())
        for segment in shard["segments"]:
            if segment["name"] in seen_names:
                raise ValueError("Duplicate segment identity across shards")
            seen_names.add(segment["name"])
    for shard in shards:
        with np.load(shard["processed_path"], allow_pickle=False) as archive:
            expected = {s["name"]+suffix for s in shard["segments"] for suffix in ("", "_phase")}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise ValueError("Shard array coverage differs")
            for segment in shard["segments"]:
                name = segment["name"]
                block, phase = archive[name], archive[name+"_phase"]
                if (block.shape != (segment["hours"], 1, 4) or phase.shape != (segment["hours"],)
                        or len(phase) < 193 or not np.issubdtype(phase.dtype, np.integer)
                        or not np.all(np.diff(phase) == 1)
                        or int(phase[0]) != segment["start_hour"] or int(phase[-1]) != segment["end_hour"]
                        or not np.isnan(block[:, 0, 1:]).all()):
                    raise ValueError("Shard temporal/physical-channel contract differs")
                values = block[:, 0, 0].astype(np.float64)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Finite nonnegative prefix values required")
                yield segment["entity"], values, phase.astype(np.int64)
