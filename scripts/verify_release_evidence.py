"""Read-only release audit: pinned data, frozen artifacts, saved outputs and documentation links."""

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from src.data.io import sha256
from src.inference.freeze import config_hash, verify_artifacts
from src.utils.config import load_config


def main():
    root = Path.cwd()
    checks = []

    def record(name, passed, detail=""):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    data = load_config("configs/data.yaml")
    manifest = json.loads(Path("reports/data_manifest.json").read_text())
    record("pinned_revision", manifest["revision"] == data["revision"])
    raw = Path(data["raw_dir"])
    for name, digest in manifest["checksums"].items():
        p = raw / name
        record(
            "raw:" + name,
            p.exists() and sha256(p) == digest,
            "Local original data checksum, no training",
        )
    frozen = load_config("configs/champion.yaml")
    record(
        "canonical_panel",
        sha256(Path(data["processed_dir"]) / "panel.npz") == frozen["panel_sha256"],
    )
    tag = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "pre-holdout^{commit}"],
        text=True,
    ).strip()
    freeze = json.loads(Path("reports/freeze_manifest.json").read_text())
    record("real_pre_holdout_tag", tag == freeze["commit"])
    record("frozen_config_semantic_hash", config_hash(frozen) == freeze["config_sha256"])
    for name, digest in freeze["protected_sources"].items():
        tagged = (
            subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "show", f"pre-holdout:{name}"]
            )
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        import hashlib

        record("tagged_source:" + name, hashlib.sha256(tagged.encode()).hexdigest() == digest)
    artifacts = verify_artifacts()
    record("champion_artifact_manifest", artifacts["freeze_commit"] == tag)
    done = json.loads(Path("reports/holdout_completion.json").read_text())
    record(
        "final_completion_identity",
        done["freeze_commit"] == tag and done["frozen_config_sha256"] == config_hash(frozen),
    )
    for name, digest in done["files"].items():
        record("saved_holdout:" + name, sha256(Path(name)) == digest)
    # Inspect local Markdown file references only, excluding fenced examples and network URLs.
    broken = []
    paths = [Path("README.md"), *Path("docs").rglob("*.md")]
    for path in paths:
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            target = unquote(target)
            if not (path.parent / target).exists():
                broken.append({"document": path.as_posix(), "target": target})
    record("local_document_links", not broken, json.dumps(broken))
    result = {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "scope": "Pinned inputs and frozen outputs checked without training or final replay; local doc targets only, no remote link availability claim",
        "checks": checks,
    }
    Path("reports/release_evidence_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print("Release evidence:", result["status"], ";", len(checks), "checks")
    for check in checks:
        if check["status"] == "FAIL":
            print(check)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
