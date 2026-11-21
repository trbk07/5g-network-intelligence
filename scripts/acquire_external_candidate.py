"""Download a revision-pinned external dataset candidate for audit, not automatic model training."""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


def main():
    repo = "CTTCADAPTRU/OAI-KPM-dataset"
    commit = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/commits/main"], text=True, encoding="utf-8"
        )
    )["sha"]
    tree = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/git/trees/{commit}?recursive=1"],
            text=True,
            encoding="utf-8",
        )
    )
    destination = Path("data/external/oai_kpm") / commit
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for entry in tree["tree"]:
        name = entry["path"]
        if entry["type"] != "blob" or not (
            name.lower().endswith((".xlsx", ".csv", ".md")) or name == "LICENSE"
        ):
            continue
        if entry.get("size", 0) > 100_000_000:
            raise ValueError("Candidate file exceeds preflight download limit")
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe upstream filename")
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/" + urllib.parse.quote(name)
        if not target.exists():
            with urllib.request.urlopen(url, timeout=90) as response, target.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        files.append(
            {"path": name, "bytes": target.stat().st_size, "sha256": digest, "source_url": url}
        )
    result = {
        "repo": repo,
        "revision": commit,
        "files": files,
        "training_performed": False,
        "status": "downloaded_for_schema_license_sampling_audit_only",
        "known_domain": "OAI emulated RAN, not original commercial beam network",
        "do_not_claim": "Additional data alone does not establish improved original-network WAPE",
    }
    Path("reports/external_candidate_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print("Downloaded pinned external candidate:", commit, "files:", len(files))


if __name__ == "__main__":
    main()
