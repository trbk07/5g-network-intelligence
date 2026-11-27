"""Acquire pinned, licensed safetensors weights; never instantiate or fit a model."""

import hashlib
import json
from pathlib import Path
import urllib.request

from src.utils.config import load_config
from src.data.io import sha256


def main():
    cfg = load_config("configs/chronos2_synth.yaml")
    repo, revision = cfg["model_id"], cfg["revision"]
    metadata_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != revision or metadata.get("cardData", {}).get("license") != cfg["license"]:
        raise ValueError("Pinned model revision/license differs")
    weight = next(item for item in metadata["siblings"] if item["rfilename"] == cfg["weights_file"])
    if weight["lfs"]["sha256"] != cfg["weights_sha256"] or weight["size"] != cfg["weights_bytes"]:
        raise ValueError("Publisher weight checksum or byte count differs")
    root = Path("data/external/pretrained/chronos2_synth")/revision
    root.mkdir(parents=True, exist_ok=True)
    (root/"publisher_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = {}
    for name in ("README.md", "config.json", cfg["weights_file"]):
        destination = root/name
        if name == cfg["weights_file"] and destination.exists():
            if destination.stat().st_size == cfg["weights_bytes"] and sha256(destination) == cfg["weights_sha256"]:
                files[name] = {"sha256": cfg["weights_sha256"], "bytes": destination.stat().st_size}
                continue
            raise ValueError("Existing pinned weight file is corrupt; preserve it for inspection")
        temporary = destination.with_suffix(destination.suffix+".part")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        digest, size, last_progress = hashlib.sha256(), 0, 0
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(8*1024*1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size-last_progress >= 64*1024*1024:
                    print(f"{name}: downloaded {size/1024**2:.0f} MiB", flush=True)
                    last_progress = size
        if name == cfg["weights_file"] and (digest.hexdigest() != cfg["weights_sha256"] or size != cfg["weights_bytes"]):
            raise ValueError("Downloaded safetensors checksum/size differs; unverified .part retained")
        if name == "config.json" and digest.hexdigest() != cfg["config_sha256"]:
            raise ValueError("Pinned model configuration checksum differs")
        temporary.replace(destination)
        files[name] = {"sha256": digest.hexdigest(), "bytes": size}
    report = {"model_id": repo, "revision": revision, "license": cfg["license"],
              "model_card": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
              "files": files, "local_directory": root.as_posix(),
              "publisher_states_synthetic_only_pretraining": True,
              "independent_corpus_audit_performed": False,
              "training_performed": False, "model_instantiated_locally": False,
              "holdout_accessed": False}
    output = Path("reports/chronos2_synth_acquisition.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
