"""Run provenance without machine-specific paths or secrets."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def collect_run_metadata(config: dict) -> dict:
    """Return config hash, environment, seed, revision and current Git commit."""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "PyYAML")},
        "seed": config.get("seed"),
        "dataset_revision": config.get("revision"),
    }
