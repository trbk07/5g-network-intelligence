"""Export completed Drive results as regular archive files; never repeat fitting."""

import hashlib
import json
from pathlib import Path
import tarfile


def main():
    """Package the completed immutable seed-29 tree and allowlisted reports."""
    root = Path("/content/drive/MyDrive/5g-network-intelligence/checkpoints/cell_seed29_96806fe4ab43")
    state = json.loads((root/"runtime_state.json").read_text())
    if state["status"] != "complete" or state.get("exit_code") != 0:
        raise RuntimeError("Wait for verified terminal training state; export cannot train")
    artifact = root/"artifacts/cell_attention_seed29"
    manifest = json.loads((artifact/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = artifact/name
        if not path.resolve().is_relative_to(artifact.resolve()) or path.is_symlink():
            raise ValueError("Unexpected artifact path/link")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Completed artifact checksum differs")
    output = Path("/content/seeded_cell_results.tar.gz")
    partial = output.with_suffix(".part")
    with tarfile.open(partial, "w:gz", dereference=True) as archive:
        archive.add(artifact, arcname="artifacts/cell_attention_seed29")
        for name in ("cell_attention_seed29_development.csv", "cell_attention_seed29_training.log"):
            archive.add(root/"reports"/name, arcname="reports/"+name)
    partial.replace(output)
    print(json.dumps({"archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "bytes": output.stat().st_size, "training_performed_by_export": False}))


if __name__ == "__main__":
    main()
