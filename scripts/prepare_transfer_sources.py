"""Prepare multiple audited domains independently; preserve source IDs and future-prefix boundaries."""

import json
from pathlib import Path

from scripts.prepare_transfer_data import prepare
from src.data.io import sha256
from src.utils.config import load_config


def main():
    base = load_config("configs/transfer_research.yaml")
    registry = load_config("configs/transfer_sources.yaml")
    reports = []
    for source in registry["sources"]:
        cfg = {
            **base,
            "external_file": source["file"],
            "external_sha256": source["sha256"],
            "external_training_fraction": registry["external_training_fraction"],
        }
        previous = Path("reports") / f"{source['slug']}_preparation.json"
        result = None
        if previous.exists():
            existing = json.loads(previous.read_text())
            prepared = Path(
                existing.get("processed_path", f"data/processed/{source['slug']}_training.npz")
            )
            if (
                existing["source_sha256"] == source["sha256"]
                and prepared.exists()
                and sha256(prepared) == existing["processed_sha256"]
            ):
                result = existing
        if result is None:
            result = prepare(cfg, slug=source["slug"])
        result["source_id"] = source["slug"]
        reports.append(result)
        Path("reports/multisource_preparation.json").write_text(
            json.dumps(
                {
                    "sources": reports,
                    "training_performed": False,
                    "future_target_use": False,
                    "pooled_training_permitted": False,
                    "remaining_gate": "Cross-source event overlap audit, then per-source/balanced-pool cloud ablations",
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(source["slug"], "prepared", flush=True)


if __name__ == "__main__":
    main()
