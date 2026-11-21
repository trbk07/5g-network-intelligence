"""Read-only workbook audit; preserve upstream file and reject incompatible training data."""

import json
from pathlib import Path

import numpy as np
import openpyxl


def main():
    manifest = json.loads(Path("reports/external_candidate_manifest.json").read_text())
    root = Path("data/external/oai_kpm") / manifest["revision"]
    path = next(root.glob("*.xlsx"))
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook["KPM Metrics"].iter_rows(values_only=True))
    times = np.asarray([r[2] for r in rows[1:] if r[2] is not None], dtype=float)
    result = {
        "source_revision": manifest["revision"],
        "license": "CC0-1.0 per upstream LICENSE",
        "sheets": {
            s.title: {"rows_including_header": s.max_row, "columns": s.max_column} for s in workbook
        },
        "kpm_headers": list(rows[0]),
        "kpm_records": len(times),
        "distinct_ues": len({r[1] for r in rows[1:] if r[1] is not None}),
        "latency_column_observation": "Values resemble epoch microseconds with approximately one-second increments; do not relabel as latency or timestamp without schema confirmation",
        "span_if_microsecond_time_seconds": float((times.max() - times.min()) / 1e6),
        "median_step_if_microseconds_seconds": float(np.median(np.diff(times)) / 1e6),
        "hourly_beam_training_eligible": False,
        "decision": "Reject direct merge/pretraining for current hourly beam forecasting pending corrected source/schema; short single-UE record contradicts five-week README claim",
        "training_performed": False,
    }
    workbook.close()
    Path("reports/oai_candidate_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print("External OAI audit: not eligible for direct hourly beam training")


if __name__ == "__main__":
    main()
