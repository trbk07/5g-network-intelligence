"""Acquire openly licensed commercial-counter partitions with upstream checksum verification."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


def main():
    record = "17815388"
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
        metadata = json.load(response)
    if metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected license; inspect source before use")
    destination = Path("data/external/zenodo") / record
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "record.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = []
    for entry in metadata["files"]:
        if entry["key"] not in ("README.md", "Dataset_01.zip", "Dataset_02.zip"):
            continue
        target = destination / entry["key"]
        if not target.exists() or target.stat().st_size != entry["size"]:
            partial = target.with_suffix(target.suffix + ".partial")
            with (
                urllib.request.urlopen(entry["links"]["self"], timeout=120) as source,
                partial.open("wb") as output,
            ):
                while block := source.read(1024 * 1024):
                    output.write(block)
            partial.replace(target)
        payload = target.read_bytes()
        if "md5:" + hashlib.md5(payload).hexdigest() != entry["checksum"]:
            raise ValueError(f"Upstream checksum mismatch: {target}")
        files.append(
            {
                "path": entry["key"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "upstream_checksum": entry["checksum"],
                "url": entry["links"]["self"],
            }
        )
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as archive:
                if sum(i.file_size for i in archive.infolist()) > 2_000_000_000:
                    raise ValueError("Uncompressed audit partition exceeds limit")
                for info in archive.infolist():
                    resolved = (destination / info.filename).resolve()
                    if (
                        not resolved.is_relative_to(destination.resolve())
                        or (info.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise ValueError("Unsafe ZIP entry")
                archive.extractall(destination)
        print("Acquired and verified", entry["key"], flush=True)
    result = {
        "record": record,
        "doi": "10.5281/zenodo.17815388",
        "license": "CC-BY-4.0",
        "creators": metadata["metadata"].get("creators", []),
        "files": files,
        "training_performed": False,
        "status": "acquired_for_compatibility_and_temporal_audit_only",
        "sampling_claim": "15-minute commercial sector counters, not original beam series",
        "not_downloaded": ["Dataset_03.zip (larger partition; inspect smaller partitions first)"],
    }
    Path("reports/zenodo_counter_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
