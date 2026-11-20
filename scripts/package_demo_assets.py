"""Package already verified demo assets for release distribution; never train."""

import json
import tarfile
from pathlib import Path

from src.data.io import sha256
from src.inference.predictor import Predictor


def package(output=".cache/release/5g-demo-assets.tar.gz"):
    predictor = Predictor()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(Path("demo_assets").rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.as_posix())
    result = {
        "filename": target.name,
        "sha256": sha256(target),
        "size_bytes": target.stat().st_size,
        "model_version": predictor.version,
        "beams_loaded": len(predictor.store.index),
        "content": "Frozen champion and 24 real NETOP GOTS-F replay beams; MIT dataset attribution in README",
        "training_performed": False,
    }
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    Path("reports/demo_asset_package.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    package()
