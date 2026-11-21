"""Memory-bounded structural audit of added multi-band files; no target statistics or fitting."""

import json
from pathlib import Path

import pandas as pd

from src.data.io import sha256


def main():
    root = Path("data/external/zenodo/17815388/Dataset_03")
    results = []
    for path in sorted(root.rglob("*NR*.csv")):
        keys = ["Base station", "Sector", "Timestamp"]
        frame = pd.read_csv(
            path, usecols=keys, dtype={"Base station": "category", "Sector": "int16"}
        )
        frame.Timestamp = pd.to_datetime(frame.Timestamp, utc=True, errors="raise")
        duplicates = int(frame.duplicated(keys).sum())
        groups = []
        for entity, group in frame.groupby(["Base station", "Sector"], observed=True):
            t = group.Timestamp.sort_values()
            gaps = t.diff().dt.total_seconds()
            segments = gaps.ne(900).cumsum()
            groups.append(
                {
                    "entity": [str(entity[0]), int(entity[1])],
                    "rows": len(t),
                    "start": str(t.iloc[0]),
                    "end": str(t.iloc[-1]),
                    "longest_exact_15m_run": int(t.groupby(segments).size().max()),
                    "duplicate_times": int(t.duplicated().sum()),
                    "gaps_over_15m": int((gaps > 900).sum()),
                }
            )
        result = {
            "file": path.relative_to(root.parent).as_posix(),
            "sha256": sha256(path),
            "rows": len(frame),
            "duplicate_keys": duplicates,
            "entity_count": len(groups),
            "entities": groups,
            "eligible_for_next_schema_audit": duplicates == 0,
            "target_values_inspected": False,
        }
        results.append(result)
        Path("reports/zenodo_large_key_audit.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8", newline="\n"
        )
        print(
            path.name,
            "rows",
            len(frame),
            "entities",
            len(groups),
            "duplicate keys",
            duplicates,
            "max continuous days",
            max(g["longest_exact_15m_run"] for g in groups) / 96,
            flush=True,
        )
        del frame


if __name__ == "__main__":
    main()
