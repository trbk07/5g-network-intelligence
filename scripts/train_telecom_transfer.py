"""Hosted three-arm Telecom data/compute ablation with separate immutable outputs."""

import json
from pathlib import Path

from scripts.configure_telecom_transfer import ARMS, protocols
from scripts.train_cell_multisource import main as run_transfer
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    expected = protocols(load_config("configs/cell_multisource_v2.yaml"))
    for arm, _, _ in ARMS:
        config = f"configs/telecom_transfer_{arm}.yaml"
        if load_config(config) != expected[arm]:
            raise ValueError("Declared data/compute protocol differs")
    ready = json.loads(Path("reports/telecom_italia_preparation.json").read_text())
    if not ready["training_ready"] or ready["physical_target_pooling_with_PRB"]:
        raise ValueError("Audited representation-only source required")
    root = Path("artifacts/telecom_transfer")
    if root.exists():
        raise FileExistsError("Inspect existing job and checkpoints before any repeat fit")
    for arm, _, _ in ARMS:
        print(json.dumps({"arm": arm, "pretraining_epochs": expected[arm]["pretraining_epochs"],
                          "source_domains": len(expected[arm]["source_reports"])}), flush=True)
        run_transfer(f"configs/telecom_transfer_{arm}.yaml", str(root/arm), f"telecom_transfer_{arm}")
    from src.data.io import sha256
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
