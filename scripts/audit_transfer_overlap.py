"""Audit prepared-source interval overlap without evaluating future targets."""

import json
from pathlib import Path

import pandas as pd

from src.data.io import sha256


def main():
    preparation = json.loads(Path("reports/multisource_preparation.json").read_text())
    sources = preparation["sources"]
    for source in sources:
        if sha256(Path(source["processed_path"])) != source["processed_sha256"]:
            raise ValueError("Prepared array checksum mismatch")
        cutoff = pd.Timestamp(source["cutoff_utc"])
        if source["future_targets_used"] or any(
            pd.Timestamp(s["end"]) > cutoff for s in source["segments"]
        ):
            raise ValueError("Prepared source includes reserved future observations")
    pairs = []
    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            by_entity = {}
            for segment in right["segments"]:
                by_entity.setdefault(tuple(segment["entity"]), []).append(segment)
            overlaps = 0
            past_other_cutoff = 0
            for segment in left["segments"]:
                for other in by_entity.get(tuple(segment["entity"]), []):
                    if max(pd.Timestamp(segment["start"]), pd.Timestamp(other["start"])) <= min(
                        pd.Timestamp(segment["end"]), pd.Timestamp(other["end"])
                    ):
                        overlaps += 1
                if tuple(segment["entity"]) in by_entity and pd.Timestamp(
                    segment["end"]
                ) > pd.Timestamp(right["cutoff_utc"]):
                    past_other_cutoff += 1
            pairs.append(
                {
                    "left": left["source_id"],
                    "right": right["source_id"],
                    "overlapping_training_intervals_same_pseudonym": overlaps,
                    "left_training_intervals_after_right_cutoff_same_pseudonym": past_other_cutoff,
                }
            )
    result = {
        "pooled_training_permitted": True,
        "external_test_claim_permitted": False,
        "source_preparation_sha256": sha256(Path("reports/multisource_preparation.json")),
        "pairs": pairs,
        "interpretation": "Overlapping bands may represent correlated training events; balance domains and do not claim independent networks or evaluate separately reserved external tests from this pooled model.",
        "original_network_calendar_identity": "Unverified; no original holdout outcomes are opened and no fresh-test claim is made.",
    }
    Path("reports/transfer_overlap_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        "Prepared source-prefix checks passed; correlated-band overlap is explicit; external-test claims disabled"
    )


if __name__ == "__main__":
    main()
