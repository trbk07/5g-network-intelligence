"""Tiny baseline-only assets for tests; statistics are allowed, no model fitting."""

import json

import numpy as np

from src.data.io import sha256
from src.data.panel import Panel, save_panel
from src.risk.thresholds import fit_thresholds


def baseline_assets(root):
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "champion"
    directory.mkdir()
    values = np.zeros((240, 2, 4))
    values[:, 0, :] = (np.arange(240) % 24 + 1)[:, None]
    panel = Panel(values, np.arange(240), ("0_0_0", "0_0_1"), {})
    save_panel(panel, root / "replay.npz")
    fit = Panel(values[:200], np.arange(200), panel.beam_ids, {})
    thresholds = fit_thresholds(
        fit,
        {
            "quantiles": [0.9, 0.95],
            "epsilon": 1e-8,
            "min_active_ratio": 0.05,
            "min_observations": 168,
        },
    )
    thresholds.save(directory / "thresholds.npz")
    meta = {
        "family": "persistence",
        "fit_end": 199,
        "frozen_config_sha256": "fixture-config",
        "freeze_commit": "fixture-commit",
        "dataset_revision": "fixture-revision",
        "beam_ids": list(panel.beam_ids),
    }
    (directory / "champion_meta.json").write_text(json.dumps(meta))
    files = {p.name: sha256(p) for p in directory.iterdir()}
    (directory / "artifacts_manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "frozen_config_sha256": "fixture-config",
                "freeze_commit": "fixture-commit",
            }
        )
    )
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    manifest = {
        "files": files,
        "beam_ids": list(panel.beam_ids),
        "fit_end": 199,
        "frozen_config_sha256": "fixture-config",
        "dataset_revision": "fixture-revision",
        "dataset_scope": "Synthetic baseline fixture; never a measured model result",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root
