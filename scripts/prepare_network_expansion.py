"""Audit licensed newly acquired prefixes and export gap-preserving hourly windows."""

import json
from pathlib import Path

import numpy as np
import tables

from scripts.prepare_cesnet_pretraining import split_segments
from src.data.io import sha256


def main():
    acquisition = json.loads(Path("reports/network_expansion_acquisition.json").read_text())
    summaries = []
    for item in acquisition["files"]:
        path = Path(item["path"])
        if path.name.startswith("TelecomItalia-"):
            continue  # The large collection needs a separate streaming audit.
        if sha256(path) != item["sha256"]:
            raise ValueError("Acquired payload checksum differs")
        vigo = path.name.startswith("Vigo-")
        subset = path.name.split("-")[2] if vigo else "agg23"
        source_id = "vigo_"+subset if vigo else "cesnet_agg23"
        report_path = Path(f"reports/{source_id}_preparation.json")
        output = Path(f"data/processed/{source_id}_training.npz")
        if output.exists():
            previous = json.loads(report_path.read_text())
            if sha256(output) != previous["processed_sha256"] or previous["source_sha256"] != item["sha256"]:
                raise ValueError("Existing preparation identity differs")
            summaries.append({"source": source_id, "segments": len(previous["segments"]), "already_verified": True})
            continue
        with tables.open_file(path, "r") as handle:
            time_node = handle.root.times.times_1_hour if vigo else handle.root.times.times_1_minute
            times = time_node[:]
            seconds = times["time"].astype(np.int64)
            interval = 3600 if vigo else 60
            if not np.array_equal(times["id_time"], np.arange(len(times))) or not np.all(np.diff(seconds) == interval):
                raise ValueError("Expected regular publisher time grid")
            cutoff = int(len(times)*.8)
            table = handle.root.network_operator.agg_1_hour if vigo else handle.root.CESNET2.agg_1_minute
            entity_key = "ts_id" if vigo else "id"
            value_key = ("sum_scaled_active_client_sessions" if subset == "sessions" else "sum_scaled_bts") if vigo else "bytes"
            keys = table.read(field="id_time")
            entities = table.read(field=entity_key)
            if len(np.unique(np.rec.fromarrays([entities, keys]))) != len(keys) or keys.max() >= len(times):
                raise ValueError("Duplicate or out-of-range entity/time keys")
            # Metric values are read only from the chronological external prefix.
            prefix = table.read_where(f"id_time < {cutoff}")
            arrays, segments = {}, []
            for entity in np.unique(prefix[entity_key]):
                rows = prefix[prefix[entity_key] == entity]
                hours = seconds[rows["id_time"]]//3600
                values = rows[value_key].astype(np.float64)
                if not vigo:
                    unique, counts = np.unique(hours, return_counts=True)
                    kept = unique[counts == 60]
                    values = np.array([values[hours == hour].sum() for hour in kept])
                    hours = kept
                for phase, signal in split_segments(hours, values):
                    name = f"segment_{len(segments)}"
                    block = np.full((len(signal), 1, 4), np.nan)
                    block[:, 0, 0] = signal
                    arrays[name], arrays[name+"_phase"] = block, phase
                    segments.append({"name": name, "entity": int(entity), "hours": len(signal),
                                     "start_hour": int(phase[0]), "end_hour": int(phase[-1])})
        if not segments:
            raise ValueError("No continuous prefix windows of at least 193 hours: " + source_id)
        np.savez_compressed(output, **arrays)
        report = {"source_id": source_id, "source_sha256": item["sha256"],
            "processed_path": output.as_posix(), "processed_sha256": sha256(output),
            "license": item["license"], "publisher": acquisition["record"],
            "cutoff_end_exclusive": cutoff, "source_grid_interval_seconds": interval,
            "signal": "publisher_scaled_traffic_or_sessions" if vigo else "bytes_per_hour",
            "publisher_per_series_scaling": vigo,
            "publisher_scaling_fit_scope": "unknown; do not claim externally future-free scaling" if vigo else "not_applicable",
            "timestamps": "publisher_shifted_relative_time" if vigo else "publisher_UTC_time",
            "segments": segments, "training_performed": False,
            "physical_target_pooling_with_PRB": False, "future_metric_values_read": False,
            "caveat": "Representation pretraining candidate only; original-domain benefit unproven"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summaries.append({"source": source_id, "segments": len(segments), "hours": sum(s["hours"] for s in segments),
                          "processed_sha256": report["processed_sha256"]})
    Path("reports/network_expansion_preparation.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
