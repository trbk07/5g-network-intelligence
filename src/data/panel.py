"""Strict, aligned time × beam × KPI arrays with explicit missing observations."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import KPIS, parse_beam_id


@dataclass
class Panel:
    """Aligned values with source-relative time indices and provenance."""

    values: np.ndarray
    time: np.ndarray
    beam_ids: tuple[str, ...]
    metadata: dict

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.time), len(self.beam_ids), len(KPIS)):
            raise ValueError("Panel shape must match time, beam and KPI axes")
        if len(set(self.beam_ids)) != len(self.beam_ids):
            raise ValueError("Duplicate beam identifiers")
        if not np.issubdtype(self.time.dtype, np.integer) or np.any(np.diff(self.time) <= 0):
            raise ValueError("Time must contain strictly increasing integer indices")
        if np.isinf(self.values).any():
            raise ValueError("Infinite KPI values are not supported")


def build_panel(files: dict[str, str | Path], metadata: dict | None = None) -> Panel:
    """Outer-align four wide CSVs; reject duplicate keys and malformed numbers."""
    if set(files) != set(KPIS):
        raise ValueError(f"Exactly these KPIs are required: {KPIS}")
    frames = {}
    source_audit = {}
    for kpi in KPIS:
        path = Path(files[kpi])
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        # Actual release uses an unnamed first index column rather than card's Time.
        unnamed_index = bool(header and header[0] == "" and "Time" not in header)
        if unnamed_index:
            header[0] = "Time"
        if len(header) != len(set(header)) or header.count("Time") != 1:
            raise ValueError(f"Duplicate columns or missing Time: {path}")
        frame = pd.read_csv(path)
        if unnamed_index:
            frame = frame.rename(columns={frame.columns[0]: "Time"})
        frame = frame.set_index("Time")
        numeric_time = pd.to_numeric(frame.index, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric_time).all() or np.any(numeric_time % 1):
            raise ValueError("Time must be finite integer indices")
        frame.index = numeric_time.astype(np.int64)
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate Time keys: {path}")
        for beam in frame.columns:
            parse_beam_id(beam)
        if not len(frame) or not len(frame.columns):
            raise ValueError(f"Empty KPI file: {path}")
        frame = frame.apply(pd.to_numeric, errors="raise")
        frames[kpi] = frame
        source_audit[kpi] = {"rows": len(frame), "beams": len(frame.columns),
                             "unnamed_time_column": unnamed_index,
                             "duplicate_time_keys": 0, "duplicate_beam_columns": 0}
    beams = tuple(sorted(set().union(*(f.columns for f in frames.values())), key=parse_beam_id))
    time = np.array(sorted(set().union(*(f.index for f in frames.values()))), dtype=np.int64)
    values = np.stack([frames[k].reindex(index=time, columns=beams).to_numpy(dtype=np.float64)
                       for k in KPIS], axis=-1)
    return Panel(values, time, beams, {**(metadata or {}), "source_audit": source_audit,
                                      "kpis": list(KPIS)})


def save_panel(panel: Panel, path: str | Path) -> None:
    """Save a pickle-free NPZ and adjacent human-readable metadata JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=panel.values, time=panel.time,
                        beam_ids=np.array(panel.beam_ids),
                        metadata=np.array(json.dumps(panel.metadata)))
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps({**panel.metadata, "shape": list(panel.values.shape)}, indent=2),
        encoding="utf-8")


def load_panel(path: str | Path) -> Panel:
    """Load and validate a saved panel without pickle."""
    with np.load(path, allow_pickle=False) as data:
        return Panel(data["values"], data["time"], tuple(data["beam_ids"].tolist()),
                     json.loads(str(data["metadata"])))
