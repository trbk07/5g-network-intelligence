"""Freeze attestation and one-run guard checked before accessing holdout targets."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from src.data.io import sha256
from src.utils.config import load_config


def config_hash(config: dict) -> str:
    """Hash parsed configuration, independent of Windows/Linux line endings."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hash(path: Path) -> str:
    """Canonical UTF-8/LF hash for frozen code/config text."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    """Run read-only Git inspection in this project, tolerating managed-account ownership."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", *args], text=True
    ).strip()


def create_attestation(path: str = "reports/freeze_manifest.json") -> dict:
    """Require a real pre-holdout Git tag at HEAD; record all protected source hashes."""
    commit = git("rev-parse", "pre-holdout^{commit}")
    if commit != git("rev-parse", "HEAD"):
        raise RuntimeError("Tag pre-holdout must point at HEAD")
    config = load_config("configs/champion.yaml")
    if not config.get("frozen"):
        raise RuntimeError("Champion is not frozen")
    protected = [Path("pyproject.toml"), Path("plan5G.md")]
    for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".yaml")):
        protected.extend(
            p for p in Path(folder).rglob(f"*{suffix}") if "__pycache__" not in p.parts
        )
    hashes = {}
    for p in protected:
        tagged = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={Path.cwd().as_posix()}",
                "show",
                f"pre-holdout:{p.as_posix()}",
            ]
        ).decode("utf-8")
        if tagged.replace("\r\n", "\n") != p.read_text(encoding="utf-8"):
            raise RuntimeError(f"Working source differs from frozen tag: {p}")
        hashes[p.as_posix()] = source_hash(p)
    result = {
        "tag": "pre-holdout",
        "commit": commit,
        "config_sha256": config_hash(config),
        "protected_sources": hashes,
        "git_tag_verified": True,
        "verification": "Created from actual local Git tag; cloud verifies packaged commit and source/config hashes.",
    }
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    return result


def verify_frozen(
    frozen: str = "configs/champion.yaml", attestation: str = "reports/freeze_manifest.json"
) -> tuple[dict, dict]:
    """Reject missing tag attestation, altered config/source or wrong executing commit."""
    config = load_config(frozen)
    manifest = json.loads(Path(attestation).read_text(encoding="utf-8"))
    if not config.get("frozen") or not config.get("holdout_allowed"):
        raise RuntimeError("Final holdout is not authorized by a frozen config")
    if manifest.get("tag") != "pre-holdout" or not manifest.get("git_tag_verified"):
        raise RuntimeError("Missing verified pre-holdout tag")
    if config_hash(config) != manifest["config_sha256"]:
        raise RuntimeError("Frozen config hash mismatch")
    commit = os.environ.get("SOURCE_GIT_COMMIT")
    if commit is None:
        commit = git("rev-parse", "HEAD")
        if git("rev-parse", "pre-holdout^{commit}") != manifest["commit"]:
            raise RuntimeError("Git tag commit mismatch")
    if commit != manifest["commit"]:
        raise RuntimeError("Executing commit differs from pre-holdout tag")
    for name, digest in manifest["protected_sources"].items():
        if source_hash(Path(name)) != digest:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    return config, manifest


def verify_artifacts(directory: str | Path = "artifacts/champion") -> dict:
    """Require every declared final artifact checksum to match before inference/evaluation."""
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise RuntimeError(f"Final artifact mismatch: {name}")
    return manifest


def claim_holdout(path: str | Path, frozen_hash: str) -> None:
    """Atomically reserve a single attempt; failed attempts remain visible and require an audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"status": "started", "frozen_config_sha256": frozen_hash}, handle)
    except FileExistsError as exc:
        raise RuntimeError("Holdout attempt already exists; refuse automatic rerun") from exc
