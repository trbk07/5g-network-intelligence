"""Stream Telecom Italia prefixes into shards; no fitting or PRB target pooling."""

import json
from pathlib import Path

import numpy as np
import tables

from scripts.prepare_cesnet_pretraining import split_segments
from src.data.io import sha256


def prefix_stop(keys: np.ndarray, cutoff: int, grid_size: int) -> int:
    """Validate chronological entity keys before reading any metric values."""
    if keys.ndim != 1 or not len(keys) or keys[0] < 0 or keys[-1] >= grid_size:
        raise ValueError("Invalid entity time keys")
    if np.any(np.diff(keys.astype(np.int64)) <= 0):
        raise ValueError("Duplicate or unsorted entity keys")
    return int(np.searchsorted(keys, cutoff))


def main() -> None:
    """Preserve gaps and the first 80% of the external time grid only."""
    acquisition = json.loads(Path("reports/network_expansion_acquisition.json").read_text())
    item = next(x for x in acquisition["files"] if Path(x["path"]).name.startswith("TelecomItalia-"))
    path = Path(item["path"])
    if item["license"] != "ODbL-1.0" or sha256(path) != item["sha256"]:
        raise ValueError("Source identity/license differs")
    destination = Path("data/processed/telecom_italia_prefix")
    if destination.exists():
        raise FileExistsError("Inspect previous output; never silently overwrite preparation")
    destination.mkdir(parents=True)
    shards, arrays, segments = [], {}, []
    total_hours = total_segments = skipped = 0

    def flush() -> None:
        nonlocal arrays, segments
        if not segments:
            return
        output = destination / f"shard_{len(shards):04d}.npz"
        np.savez_compressed(output, **arrays)
        shards.append({"processed_path": output.as_posix(), "processed_sha256": sha256(output),
                       "segments": segments})
        arrays, segments = {}, []

    with tables.open_file(path, "r") as handle:
        times = handle.root.times.times_1_hour[:]
        seconds = times["time"].astype(np.int64)
        if not np.array_equal(times["id_time"], np.arange(len(times))) or not np.all(np.diff(seconds) == 3600):
            raise ValueError("Unexpected hourly time grid")
        cutoff = int(.8 * len(times))
        table = handle.root.service_provider.agg_1_hour
        ranges = handle.root.service_provider.id_ranges_agg_1_hour[:]
        if (len(np.unique(ranges["ts_id"])) != len(ranges) or ranges["start"][0] != 0
                or ranges["end"][-1] != table.nrows
                or not np.array_equal(ranges["start"][1:], ranges["end"][:-1])
                or np.any(ranges["end"] <= ranges["start"])):
            raise ValueError("Entity ranges must partition the table exactly")
        for number, record in enumerate(ranges):
            start, end, entity = (int(record[k]) for k in ("start", "end", "ts_id"))
            keys = table.read(start, end, field="id_time")
            ids = table.read(start, end, field="ts_id")
            if not np.all(ids == entity):
                raise ValueError("Range crosses entity boundaries")
            count = prefix_stop(keys, cutoff, len(times))
            # Read only the physical metric prefix; later metric values remain untouched.
            values = table.read(start, start + count, field="sum_internet_traffic")
            accepted = split_segments(seconds[keys[:count]] // 3600, values)
            if not accepted:
                skipped += 1
            for hours, signal in accepted:
                name = f"segment_{total_segments}"
                block = np.full((len(signal), 1, 4), np.nan)
                block[:, 0, 0] = signal
                arrays[name], arrays[name + "_phase"] = block, hours
                segments.append({"name": name, "entity": entity, "hours": len(signal),
                                 "start_hour": int(hours[0]), "end_hour": int(hours[-1])})
                total_segments += 1
                total_hours += len(signal)
            if (number + 1) % 128 == 0:
                flush()
            if (number + 1) % 2048 == 0:
                print(json.dumps({"entities_audited": number + 1, "eligible_hours": total_hours}), flush=True)
        flush()
    report = {"source_id": "telecom_italia", "source_sha256": item["sha256"],
              "publisher": acquisition["record"], "license": item["license"],
              "attribution": item["attribution"], "cutoff_end_exclusive": cutoff,
              "signal": "publisher_sum_internet_traffic", "physical_unit_verified": False,
              "entities_audited": len(ranges), "entities_without_eligible_segments": skipped,
              "eligible_segments": total_segments, "eligible_hours": total_hours, "shards": shards,
              "future_metric_values_read": False, "training_performed": False,
              "physical_target_pooling_with_PRB": False, "training_ready": False,
              "limitations": "Requires source-overlap audit and sharded sampler integration before hosted pretraining. Country/grid entities are not independent networks. ODbL obligations apply to derived database distribution."}
    Path("reports/telecom_italia_prefix_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
