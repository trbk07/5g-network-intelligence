"""Audit external hourly traffic keys and its training prefix; never fit models."""

import json
from pathlib import Path

import numpy as np
import tables


def main():
    source = Path("data/external/cesnet/23296272/CESNET-TimeSeries24-institutions-hour.h5")
    with tables.open_file(source, "r") as handle:
        times = handle.root.times.times_1_hour[:]
        # TimeType columns require native PyTables support, without object deserialization.
        assert np.array_equal(times["id_time"], np.arange(len(times)))
        assert np.all(np.diff(times["time"].astype(np.int64)) == 3600)
        cutoff = int(len(times) * 0.8)
        table = handle.root.institutions.agg_1_hour
        keys = table.read(field="id_time")
        entities = table.read(field="id_institution")
        pairs = np.rec.fromarrays([entities, keys])
        assert len(np.unique(pairs)) == len(pairs), "Duplicate entity/hour keys"
        assert keys.max() < len(times)
        # Value diagnostics touch only the chronological external training prefix.
        prefix = table.read_where(f"id_time < {cutoff}")
        values = prefix["n_bytes"]
        segments = []
        for entity in np.unique(prefix["id_institution"]):
            hours = np.sort(prefix["id_time"][prefix["id_institution"] == entity])
            runs = np.split(hours, np.flatnonzero(np.diff(hours.astype(np.int64)) != 1) + 1)
            segments.extend(len(run) for run in runs if len(run) >= 193)
        result = {
            "source": str(source), "license": "CC-BY-4.0",
            "publisher": "https://zenodo.org/records/23296272",
            "independent_collection": "CESNET-TimeSeries24",
            "total_rows": len(keys), "entities": len(np.unique(entities)),
            "hourly_grid_length": len(times), "duplicate_keys": 0,
            "external_prefix_end_exclusive": cutoff,
            "external_future_values_inspected": False,
            "prefix_rows": len(prefix), "prefix_negative_n_bytes": int((values < 0).sum()),
            "prefix_zero_n_bytes": int((values == 0).sum()),
            "eligible_contiguous_segments": len(segments),
            "eligible_segment_hours": sum(segments),
            "minimum_segment_hours": 193,
            "candidate_signal": "n_bytes per hour; IP traffic, not PRB utilization",
            "permitted_use": "external representation pretraining with matched target-only control",
            "training_executed": False, "original_holdout_accessed": False,
        }
    output = Path("reports/cesnet_prefix_audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
