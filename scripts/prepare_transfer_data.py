"""Prepare only the external training prefix; no windows across missing hours."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.utils.config import load_config


def entity_identity(entity):
    """Pandas group keys may be NumPy scalars on cloud; persist portable JSON types."""
    return [str(entity[0]), int(entity[1])]


def prepare():
    cfg = load_config("configs/transfer_research.yaml")
    root = Path("data/external/zenodo") / cfg["external_record"]
    path = next(root.rglob(cfg["external_file"]))
    if sha256(path) != cfg["external_sha256"]:
        raise ValueError("External source differs from audited pinned CSV")
    frame = pd.read_csv(path, usecols=["Base station", "Sector", "Timestamp", "5G RB utilization"])
    frame["Timestamp"] = pd.to_datetime(frame.Timestamp, utc=True, errors="raise")
    if frame.duplicated(["Base station", "Sector", "Timestamp"]).any():
        raise ValueError("Ambiguous external natural key; refuse first-row/mean conflict handling")
    # Determine chronological boundary from timestamps only; never inspect future target quality.
    times = sorted(frame.Timestamp.unique())
    cutoff = times[int(len(times) * cfg["external_training_fraction"]) - 1]
    prefix = frame.loc[frame.Timestamp <= cutoff].copy()
    segments = []
    arrays = {}
    for entity, group in prefix.groupby(["Base station", "Sector"]):
        group = group.sort_values("Timestamp")
        hour = group.Timestamp.dt.floor("h")
        hourly = group.groupby(hour)["5G RB utilization"].agg(["mean", "count"])
        # Exact source timing and finite count are both needed; partial hours are not rescaled.
        times_per_hour = group.groupby(hour).Timestamp.apply(
            lambda s: sorted((s - s.dt.floor("h")).dt.total_seconds().tolist())
        )
        valid = (hourly["count"] == 4) & times_per_hour.apply(
            lambda offsets: offsets == [0, 900, 1800, 2700]
        )
        hourly = hourly.loc[valid]
        blocks = hourly.index.to_series().diff().dt.total_seconds().ne(3600).cumsum()
        for _, block in hourly.groupby(blocks):
            if len(block) < 193:  # 168 observed hours plus at least 24h target coverage.
                continue
            name = f"segment_{len(segments)}"
            values = np.full((len(block), 1, 4), np.nan)
            values[:, 0, 0] = block["mean"].to_numpy()
            arrays[name] = values
            arrays[name + "_phase"] = np.array([int(t.timestamp() // 3600) for t in block.index])
            segments.append(
                {
                    "name": name,
                    "entity": entity_identity(entity),
                    "hours": len(block),
                    "start": str(block.index[0]),
                    "end": str(block.index[-1]),
                }
            )
    if not segments:
        raise RuntimeError("No sufficiently long continuous external training segment")
    target = Path("data/processed/transfer_training.npz")
    target.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(target, **arrays)
    report = {
        "source_sha256": sha256(path),
        "source_file": cfg["external_file"],
        "license": "CC-BY-4.0",
        "cutoff_utc": str(cutoff),
        "future_targets_used": False,
        "segments": segments,
        "training_rows": sum(s["hours"] for s in segments),
        "aggregation": cfg["external_aggregation"],
        "unit_policy": "No physical unit equivalence claimed; relative target/history shape transfer only",
        "processed_sha256": sha256(target),
    }
    Path("reports/transfer_preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        "External training prefix:",
        len(segments),
        "continuous segments;",
        report["training_rows"],
        "entity-hours",
    )
    return report


if __name__ == "__main__":
    prepare()
