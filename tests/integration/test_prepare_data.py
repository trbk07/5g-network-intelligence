"""Offline end-to-end validation of Week 1 artifacts and split consistency."""
import json
import sys

import numpy as np
import pandas as pd
import yaml

from scripts.prepare_data import main
from src.data.io import sha256
from src.data.panel import load_panel
from src.data.schema import KPIS, SEGMENTS, csv_path


def test_panel_and_audit_cli(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    checksums = {}
    for segment in SEGMENTS:
        n = 840 if segment == "train" else 168
        for kpi in KPIS:
            name = csv_path(kpi, segment)
            path = raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({"Time": np.arange(n), "10_0_0": np.ones(n),
                                  "2_0_0": np.zeros(n)})
            frame.loc[0, "10_0_0"] = np.nan
            frame.to_csv(path, index=False)
            checksums[name] = sha256(path)
    (raw / "provenance.json").write_text(json.dumps({
        "revision": "fixture", "license": "test-only", "checksums": checksums}))
    config = {"raw_dir": str(raw), "processed_dir": str(tmp_path / "processed"),
              "reports_dir": str(tmp_path / "reports"), "seed": 17,
              "repo_id": "fixture", "revision": "fixture",
              "evaluation_config": str(tmp_path / "evaluation.yaml"),
              "continuity_verified": False, "continuity_evidence": ""}
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config))
    for step in ("panel", "audit"):
        monkeypatch.setattr(sys, "argv", ["prepare_data", "--config", str(config_path),
                                         "--step", step])
        main()
    panel = load_panel(tmp_path / "processed/panel.npz")
    assert panel.values.shape == (840, 2, 4)
    assert panel.beam_ids == ("2_0_0", "10_0_0")
    assert np.isnan(panel.values[0, 1]).all()
    split = json.loads((tmp_path / "processed/split.json").read_text())
    evaluation = yaml.safe_load((tmp_path / "evaluation.yaml").read_text())
    for key in ("fit", "validation", "holdout"):
        assert evaluation[key] == split[key]
    assert evaluation["split_mode"] == split["mode"] == "fallback"
    audit = json.loads((tmp_path / "reports/data_audit.json").read_text())
    assert audit["zeros"]["DLPRB"]["zero_ratio"] == 840 / 1679
    assert (tmp_path / "reports/figures/data_audit.png").stat().st_size > 0
    assert load_panel(tmp_path / "processed/week11.npz").values.shape == (168, 2, 4)
