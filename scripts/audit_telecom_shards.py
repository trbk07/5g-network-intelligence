"""Finalize prefix shard quality/identity audit without fitting any model."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.io import sha256


def segment_digest(values, phase):
    """Fingerprint an entire aligned segment, not a certificate of all subwindows."""
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    digest.update(np.asarray(phase, dtype="<i8").tobytes())
    return digest.hexdigest()


def main():
    report_path = Path("reports/telecom_italia_prefix_preparation.json")
    original = json.loads(report_path.read_text())
    if (original["license"] != "ODbL-1.0" or original["future_metric_values_read"]
            or original["training_performed"] or original["entities_audited"] != 29677):
        raise ValueError("Completed acquisition/prefix evidence required")
    output = Path("reports/telecom_italia_preparation.json")
    manifest_path = Path("data/processed/telecom_italia_prefix/audited_manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Never overwrite audited source identity")
    # This proves exact full-segment duplication only. It does not certify
    # different-length interval overlap or calendar identity of GOTS-F.
    seen = set()
    for other in ["cesnet", "pm01_nr1800", "pm03_nr2100", "pm03_nr3500", "vigo_sessions",
                  "vigo_vpn", "vigo_internet", "vigo_downstream", "cesnet_agg23"]:
        prior = json.loads(Path(f"reports/{other}_preparation.json").read_text())
        if sha256(Path(prior["processed_path"])) != prior["processed_sha256"]:
            raise ValueError("Existing source identity differs")
        with np.load(prior["processed_path"], allow_pickle=False) as archive:
            for segment in prior["segments"]:
                name = segment["name"]
                seen.add(segment_digest(archive[name][:, 0, 0], archive[name+"_phase"]))
    shards, names, hours = [], set(), 0
    counts = {"retained_segments": 0, "constant_segments_excluded": 0, "exact_duplicate_segments_excluded": 0}
    minimum, maximum = None, None
    for shard in original["shards"]:
        path = Path(shard["processed_path"])
        if sha256(path) != shard["processed_sha256"]:
            raise ValueError("Shard checksum mismatch")
        retained, excluded = [], []
        with np.load(path, allow_pickle=False) as archive:
            expected = {s["name"]+suffix for s in shard["segments"] for suffix in ("", "_phase")}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise ValueError("Source shard array coverage differs")
            for segment in shard["segments"]:
                name = segment["name"]
                if name in names:
                    raise ValueError("Duplicate segment identity")
                names.add(name)
                block, phase = archive[name], archive[name+"_phase"]
                if (block.shape != (segment["hours"], 1, 4) or len(phase) != len(block)
                        or len(phase) < 193 or not np.all(np.diff(phase) == 1)
                        or not np.isnan(block[:, 0, 1:]).all()
                        or int(phase[0]) != segment["start_hour"] or int(phase[-1]) != segment["end_hour"]):
                    raise ValueError("Invalid continuous source-only representation")
                values = block[:, 0, 0]
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Invalid source metric")
                # Exclude only whole constant source segments, never zero observations
                # within a varying segment or any original target observations.
                reason = "constant" if np.ptp(values) == 0 else None
                fingerprint = segment_digest(values, phase)
                if reason is None and fingerprint in seen:
                    reason = "exact_duplicate"
                if reason:
                    excluded.append({"name": name, "reason": reason})
                    counts[reason+"_segments_excluded"] += 1
                    continue
                seen.add(fingerprint)
                retained.append(segment)
                counts["retained_segments"] += 1
                hours += len(values)
                minimum = int(phase[0]) if minimum is None else min(minimum, int(phase[0]))
                maximum = int(phase[-1]) if maximum is None else max(maximum, int(phase[-1]))
        shards.append({**shard, "segments": retained, "excluded_segments": excluded})
    if len(names) != original["eligible_segments"] or not counts["retained_segments"]:
        raise ValueError("Incomplete source coverage or no informative prefix segments")
    manifest_path.write_text(json.dumps({"shards": shards}, indent=2), encoding="utf-8")
    result = {"source_id": "telecom_italia", "format": "sharded_prefix_v1", "license": original["license"],
              "attribution": original["attribution"], "publisher": original["publisher"],
              "source_sha256": original["source_sha256"], "prefix_report_sha256": sha256(report_path),
              "processed_path": manifest_path.as_posix(), "processed_sha256": sha256(manifest_path),
              "cutoff_end_exclusive": original["cutoff_end_exclusive"], "training_ready": True,
              "future_metric_values_read": False, "training_performed": False,
              "physical_target_pooling_with_PRB": False, "eligible_hours": hours, **counts,
              "retained_start_hour": minimum, "retained_end_hour": maximum,
              "sampling_domain_count": 1, "independent_network_count_claim": False,
              "original_network_calendar_identity": "unknown; no independent-test claim",
              "overlap_audit_scope": "exact entire aligned segment hashes against nine prepared domains and within collection; partial intervals not certified absent",
              "usage_scope": "context-normalized representation pretraining, reused development screening only",
              "distribution": "ODbL attribution/share-alike obligations apply to derived databases; no public raw-data push"}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
