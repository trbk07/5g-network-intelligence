"""Export audited CESNET prefix segments for hosted representation pretraining."""

import json
from pathlib import Path

import numpy as np
import tables

from src.data.io import sha256


def split_segments(hours, values, minimum=193):
    """Reject ambiguous keys and preserve hourly gaps instead of interpolating."""
    hours, values = np.asarray(hours), np.asarray(values)
    if hours.ndim != 1 or values.shape != hours.shape or minimum < 193:
        raise ValueError("Aligned hourly vectors and at least 193 observations required")
    order = np.argsort(hours)
    hours, values = hours[order], values[order]
    if len(hours) and np.any(np.diff(hours) <= 0):
        raise ValueError("Duplicate hourly keys")
    valid = np.isfinite(values) & (values >= 0)
    hours, values = hours[valid], values[valid]
    bounds = np.r_[0, np.flatnonzero(np.diff(hours) != 1) + 1, len(hours)]
    return [(hours[a:b], values[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b-a >= minimum]


def main():
    audit = json.loads(Path("reports/cesnet_prefix_audit.json").read_text())
    acquisition = json.loads(Path("reports/cesnet_acquisition.json").read_text())
    source = Path(audit["source"])
    # Pin the verified publisher payload, not merely its filename or a stale audit.
    expected = "1b9ecc4b08e768fa7374d064d96e3fd626640487f1973e2d248f7d02cb91a838"
    if sha256(source) != expected:
        raise ValueError("CESNET payload differs from verified acquisition")
    cutoff = audit["external_prefix_end_exclusive"]
    arrays, segments = {}, []
    with tables.open_file(source, "r") as handle:
        times = handle.root.times.times_1_hour[:]
        prefix = handle.root.institutions.agg_1_hour.read_where(f"id_time < {cutoff}")
        for entity in np.unique(prefix["id_institution"]):
            rows = prefix[prefix["id_institution"] == entity]
            for keys, signal in split_segments(rows["id_time"].astype(np.int64), rows["n_bytes"]):
                name = f"segment_{len(segments)}"
                # Preserve the shared four-channel container; only channel zero is observed.
                values = np.full((len(keys), 1, 4), np.nan, dtype=np.float64)
                values[:, 0, 0] = signal
                arrays[name] = values
                arrays[name + "_phase"] = times["time"][keys].astype(np.int64) // 3600
                segments.append({"name": name, "entity": int(entity), "hours": len(keys),
                                 "start_id": int(keys[0]), "end_id": int(keys[-1])})
    if sum(s["hours"] for s in segments) != audit["eligible_segment_hours"]:
        raise ValueError("Preparation differs from audited eligible coverage")
    output = Path("data/processed/cesnet_training.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Refusing to overwrite prepared source")
    np.savez_compressed(output, **arrays)
    report = {"source_sha256": expected, "processed_path": output.as_posix(),
              "processed_sha256": sha256(output), "source_id": "cesnet_timeseries24",
              "license": acquisition.get("license", "CC-BY-4.0"),
              "cutoff_end_exclusive": cutoff, "signal": "n_bytes_per_hour",
              "segments": segments, "training_performed": False,
              "physical_target_pooling_with_PRB": False, "future_values_used": False}
    Path("reports/cesnet_preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"segments": len(segments), "hours": sum(s["hours"] for s in segments),
                      "processed_sha256": report["processed_sha256"]}))


if __name__ == "__main__":
    main()
