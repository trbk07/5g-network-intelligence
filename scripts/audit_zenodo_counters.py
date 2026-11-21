"""Structural/time audit only; no test-target scoring, model fitting or cross-domain pooling."""

import json
from pathlib import Path

import pandas as pd

from src.data.io import sha256


def main():
    root = Path("data/external/zenodo/17815388")
    reports = []
    for path in sorted(root.rglob("*.csv")):
        frame = pd.read_csv(path)
        required = ["Base station", "Sector", "Timestamp"]
        if any(c not in frame for c in required):
            raise ValueError("Missing entity/time schema")
        time = pd.to_datetime(frame.Timestamp, utc=True, errors="raise")
        keys = frame[required].copy()
        keys["Timestamp"] = time
        group_stats = []
        for entity, indices in frame.groupby(["Base station", "Sector"]).groups.items():
            t = time.loc[indices].sort_values()
            gaps = t.diff().dropna().dt.total_seconds()
            group_stats.append(
                {
                    "entity": list(entity),
                    "rows": len(t),
                    "start": str(t.iloc[0]),
                    "end": str(t.iloc[-1]),
                    "span_days": (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400,
                    "duplicate_times": int(t.duplicated().sum()),
                    "cadence_seconds": gaps.value_counts().head(4).to_dict(),
                    "gaps_larger_than_15m": int((gaps > 900).sum()),
                }
            )
        reports.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "duplicate_entity_times": int(keys.duplicated().sum()),
                "entities": group_stats,
                "target_column": "5G RB utilization"
                if "5G RB utilization" in frame
                else "4G RB utilization",
                "unit_status": "RB utilization requires field-dictionary/bandwidth review; no equivalence with GOTS-F DLPRB assumed",
            }
        )
    result = {
        "record": "17815388",
        "license": "CC-BY-4.0",
        "files": reports,
        "training_performed": False,
        "selection_status": "structural audit only; no evaluation metrics or final-test inspection",
        "merge_decision": "Keep separate domain. Sector/15-minute measurements differ from commercial beam/hourly inputs. Do not claim external scores improve original-network forecasts.",
    }
    Path("reports/zenodo_counter_audit.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    for report in reports:
        print(
            report["path"],
            "rows",
            report["rows"],
            "entities",
            len(report["entities"]),
            "span_days",
            [round(g["span_days"], 1) for g in report["entities"]],
            "duplicates",
            report["duplicate_entity_times"],
        )


if __name__ == "__main__":
    main()
