"""Import a completed hosted research archive without touching frozen artifacts."""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from src.data.io import sha256


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def import_results(archive_path, workspace, kind="development"):
    """Validate all input before copying an allowlist; existing different files fail."""
    workspace = Path(workspace).resolve()
    archive_path = Path(archive_path).resolve()
    stage_root = workspace / ".cache"
    stage_root.mkdir(exist_ok=True)
    if kind not in ("development", "candidate", "calibration", "temporal_linear", "development_blend"):
        raise ValueError("Unknown hurdle archive kind")
    candidate = kind == "candidate"
    prefix = PurePosixPath("artifacts/hurdle_candidate" if candidate else "artifacts/hurdle_research")
    required = {"reports/hurdle_candidate_validation.csv", "reports/forecasts/hurdle_candidate_val.npz"} if candidate else {"reports/hurdle_development_scores.csv"}
    report_names = required | {"reports/hurdle_audit_training.log" if candidate else "reports/hurdle_training.log"}
    if kind == "calibration":
        prefix = PurePosixPath("artifacts/hurdle_calibration")
        required = {"reports/hurdle_calibration_development.csv"}
        report_names = required | {"reports/hurdle_calibration.log"}
    if kind == "temporal_linear":
        prefix = PurePosixPath("artifacts/temporal_linear")
        required = {"reports/temporal_linear_development.csv"}
        report_names = required | {"reports/temporal_linear_training.log"}
    if kind == "development_blend":
        prefix = PurePosixPath("artifacts/development_blend")
        required = {"reports/development_blend_selection_grid.csv", "reports/development_blend_confirmation.csv"}
        report_names = required | {"reports/development_blend.log"}
    with tempfile.TemporaryDirectory(prefix="hurdle-import-", dir=stage_root) as temp:
        staging = Path(temp)
        with tarfile.open(archive_path, "r:gz") as archive:
            seen = set()
            for member in archive.getmembers():
                path = safe_relative(member.name)
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise ValueError("Links and special archive members are forbidden")
                if not member.isfile():
                    continue
                if path in seen:
                    raise ValueError("Duplicate archive members are ambiguous")
                seen.add(path)
                if not (path.is_relative_to(prefix) or path.as_posix() in report_names):
                    continue  # Never extract archived configs or unrelated artifacts/reports.
                destination = staging.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Unreadable archive file")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        root = staging / prefix
        manifest = json.loads((root / "manifest.json").read_text())
        declared = set()
        for name, digest in manifest["files"].items():
            relative = safe_relative(name)
            if not relative.parts or relative.as_posix() == "manifest.json":
                raise ValueError("Invalid manifest member")
            path = root.joinpath(*relative.parts)
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Research checkpoint checksum mismatch: {name}")
            declared.add(relative.as_posix())
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != declared | {"manifest.json"}:
            raise ValueError("Unmanifested research artifact")
        if not all((staging / name).is_file() for name in required):
            raise ValueError("Complete score reports and declared forecasts required")
        files = [p for p in staging.rglob("*") if p.is_file()]
        for path in files:
            destination = workspace / path.relative_to(staging)
            if destination.exists() and (not destination.is_file() or sha256(destination) != sha256(path)):
                raise ValueError(f"Refusing to overwrite different existing research output: {destination}")
        for path in files:
            destination = workspace / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
    evidence = {"archive_sha256": sha256(archive_path), "verified_artifact_files": len(declared),
                "copied_or_identical_files": len(files), "frozen_artifacts_touched": False,
                "scope": "Archive integrity only; temporal, metric and prediction verification still required"}
    evidence_name = "hurdle_candidate_import_evidence.json" if candidate else "hurdle_import_evidence.json"
    if kind == "calibration":
        evidence_name = "hurdle_calibration_import_evidence.json"
    if kind == "temporal_linear":
        evidence_name = "temporal_linear_import_evidence.json"
    if kind == "development_blend":
        evidence_name = "development_blend_import_evidence.json"
    (workspace / "reports" / evidence_name).write_text(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--kind", choices=["development", "candidate", "calibration", "temporal_linear", "development_blend"], default="development")
    args = parser.parse_args()
    print(json.dumps(import_results(args.archive, Path.cwd(), args.kind), indent=2))
