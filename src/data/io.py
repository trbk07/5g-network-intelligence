"""Revision-pinned downloads and checksummed CSV manifests."""
import csv
import hashlib
import json
from pathlib import Path

from src.data.schema import KPIS, SEGMENTS, csv_path


def sha256(path: Path) -> str:
    """Compute a streaming file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(repo_id: str, revision: str, dest: str | Path) -> dict:
    """Resolve revision once and download required CSVs; verify cached checksums."""
    from huggingface_hub import HfApi, hf_hub_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(repo_id, revision=revision)
    provenance_path = dest / "provenance.json"
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    checksums = {}
    for segment in SEGMENTS:
        for kpi in KPIS:
            name = csv_path(kpi, segment)
            target = dest / name
            if (previous.get("revision") == info.sha and target.exists()
                    and previous.get("checksums", {}).get(name) == sha256(target)):
                checksums[name] = sha256(target)
                continue
            hf_hub_download(repo_id, name, repo_type="dataset", revision=info.sha,
                            local_dir=dest, force_download=(
                                previous.get("revision") == info.sha and target.exists()))
            checksums[name] = sha256(target)
    hf_hub_download(repo_id, "README.md", repo_type="dataset", revision=info.sha, local_dir=dest)
    provenance = {"repo_id": repo_id, "revision": info.sha,
                  "license": info.card_data.to_dict().get("license", "unknown")
                  if info.card_data else "unknown",
                  "checksums": checksums}
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def build_manifest(raw_dir: str | Path) -> dict:
    """Inspect CSV row counts, columns, byte sizes and checksums."""
    raw_dir = Path(raw_dir)
    provenance = json.loads((raw_dir / "provenance.json").read_text(encoding="utf-8"))
    expected = set(provenance.get("checksums", {}))
    if not expected:
        raise ValueError("Provenance must include source file checksums")
    actual = {p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*.csv")}
    if actual != expected:
        raise ValueError(f"Source files differ: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    files = []
    for path in sorted(raw_dir.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            rows = sum(1 for _ in reader)
        digest = sha256(path)
        relative = path.relative_to(raw_dir).as_posix()
        if provenance["checksums"].get(relative) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        files.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size,
                      "rows": rows, "columns": columns})
    return {**provenance, "files": files}
