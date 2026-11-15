"""The validation runner must be independent of holdout values and reject holdout execution."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_baselines import main
from src.data.panel import Panel, save_panel
from src.data.splits import build_split_config


def test_validation_runner_holdout_canary(tmp_path, monkeypatch):
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    processed.mkdir()
    values = np.random.default_rng(17).uniform(size=(840, 3, 4))
    split = build_split_config(840)
    (processed / "split.json").write_text(json.dumps(split))
    evaluation = {k: split[k] for k in ("fit", "validation", "holdout", "horizons", "lookback")}
    evaluation["split_mode"] = split["mode"]
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation))
    config = {"processed_dir": str(processed), "reports_dir": str(reports), "seed": 17,
              "evaluation_config": str(evaluation_path), "revision": "fixture"}
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_baselines", "--config", str(path), "--segment", "val"])
    for perturb in (False, True):
        if perturb:
            values[672:] = 99999
        save_panel(Panel(values, np.arange(840), ("0_0_0", "0_0_1", "0_0_2"), {}),
                   processed / "panel.npz")
        main()
        current = pd.read_csv(reports / "baseline_benchmark_val.csv")
        if not perturb:
            original = current
        else:
            pd.testing.assert_frame_equal(original, current)
    with np.load(reports / "forecasts/baselines_val.npz", allow_pickle=False) as archive:
        assert len(json.loads(str(archive["metadata"]))["records"]) == 9
        assert max(archive["record_0_targets"]) == 671
    monkeypatch.setattr(sys, "argv", ["run_baselines", "--segment", "holdout"])
    with pytest.raises(SystemExit):
        main()
