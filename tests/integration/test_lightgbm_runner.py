"""Small end-to-end training/reload/report check; real data benchmark is separate."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.evaluate_models import main as evaluate_main
from scripts.run_baselines import main as baselines_main
from scripts.train_lgbm import main as train_main
from src.data.panel import Panel, save_panel
from src.data.splits import build_split_config

pytestmark = pytest.mark.cloud_training


def test_train_reload_report_and_holdout_independence(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    values = np.random.default_rng(17).uniform(size=(840, 3, 4))
    panel = Panel(values, np.arange(840), ("0_0_0", "0_0_1", "0_0_2"), {})
    save_panel(panel, processed / "panel.npz")
    split = build_split_config(840)
    (processed / "split.json").write_text(json.dumps(split))
    evaluation = {k: split[k] for k in ("fit", "validation", "holdout", "horizons", "lookback")}
    evaluation["split_mode"] = split["mode"]
    eval_path = tmp_path / "evaluation.yaml"
    eval_path.write_text(yaml.safe_dump(evaluation))
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump({"processed_dir": str(processed),
        "reports_dir": str(tmp_path / "reports"), "seed": 17,
        "evaluation_config": str(eval_path), "revision": "test"}))
    model_config = tmp_path / "lgbm.yaml"
    configs = {name: {"objective": objective, "num_leaves": 4, "n_estimators": 5,
                      "early_stopping_rounds": 2, "learning_rate": .1}
               for name, objective in (("c1", "regression_l1"), ("c2", "regression"))}
    model_config.write_text(yaml.safe_dump({"seed": 17, "training_stride": 2, "num_threads": 1,
        "artifact_dir": str(tmp_path / "artifacts"), "configs": configs, "selected_config": None}))
    monkeypatch.setattr(sys, "argv", ["baselines", "--config", str(data_config)])
    baselines_main()
    monkeypatch.setattr(sys, "argv", ["train", "--data-config", str(data_config),
                                     "--config", str(model_config)])
    train_main()
    argv = ["evaluate", "--model", "lightgbm", "--segment", "val", "--data-config",
            str(data_config), "--config", str(model_config)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_main()
    report_path = tmp_path / "reports/lightgbm_benchmark_val.csv"
    original = pd.read_csv(report_path)
    values[672:] = 99999
    save_panel(panel, processed / "panel.npz")
    evaluate_main()
    pd.testing.assert_frame_equal(original, pd.read_csv(report_path))
    assert len(original[original.load_tier == "All"]) == 6
    assert len(pd.read_csv(tmp_path / "reports/lightgbm_failure_examples.csv")) >= 3
    assert yaml.safe_load(model_config.read_text())["selected_config"] in ("c1", "c2")
    assert len(list((tmp_path / "artifacts").glob("*/h*/model.txt"))) == 6
    monkeypatch.setattr(sys, "argv", ["evaluate", "--model", "lightgbm", "--segment", "holdout"])
    with pytest.raises(SystemExit):
        evaluate_main()
