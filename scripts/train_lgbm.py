"""Train exactly two planned LightGBM configs on the frozen fit/validation split."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.features.static_stats import compute_static
from src.models.lgbm import load_lgbm, save_lgbm, train_lgbm
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata
from src.utils.seed import set_seed


def main() -> None:
    """Train C1/C2 × three horizons and save independent artifacts with parity checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--config", default="configs/lightgbm.yaml")
    args = parser.parse_args()
    data_config, config = load_config(args.data_config), load_config(args.config)
    if set(config["configs"]) != {"c1", "c2"}:
        raise ValueError("Week 3 permits exactly c1 and c2")
    set_seed(config["seed"])
    panel, split = validation_inputs(data_config)
    static = compute_static(panel, split["fit"][1])
    reports = Path(data_config["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "feature_manifest.json").write_text(json.dumps({
        "count": len(FEATURE_NAMES), "names": FEATURE_NAMES, "dtype": "float32",
        "row_order": "origin-major, beam-major", "fit_end": static.fit_end,
        "rolling_std_ddof": 0, "seasonal_mean": "observed-only 7 target-aligned daily lags",
        "missing_policy": "preserve NaN; native LightGBM handling",
    }, indent=2), encoding="utf-8")
    for h in split["horizons"]:
        origins = train_origins(split["fit"][1], h, stride=config["training_stride"])
        val_origins = evaluation_origins(split["fit"][1], split["validation"][1], h)
        print(f"Building +{h}h features: {len(origins)} fit / {len(val_origins)} val origins", flush=True)
        train_x = build_features(panel.values, origins, h, static)
        val_x = build_features(panel.values, val_origins, h, static)
        train_y = panel.values[origins + h, :, 0].reshape(-1)
        val_y = panel.values[val_origins + h, :, 0].reshape(-1)
        for name, parameters in config["configs"].items():
            print(f"Training {name} +{h}h...", flush=True)
            metadata = collect_run_metadata({**data_config, "parameters": parameters,
                                             "training_stride": config["training_stride"],
                                             "seed": config["seed"], "horizon": h,
                                             "split": {k: v for k, v in split.items() if k != "origins"}})
            model = train_lgbm(train_x, train_y, val_x, val_y, parameters, static, h,
                               f"lightgbm_{name}", config["seed"], config["num_threads"], metadata)
            directory = Path(config["artifact_dir"]) / name / f"h{h}"
            save_lgbm(model, directory)
            loaded = load_lgbm(directory)
            np.testing.assert_allclose(model.booster.predict(val_x[:100], num_threads=4),
                                       loaded.booster.predict(val_x[:100], num_threads=4),
                                       rtol=1e-12, atol=1e-12)
            print(f"Saved {directory}; best_iteration={model.booster.best_iteration}; parity passed",
                  flush=True)
        del train_x, val_x


if __name__ == "__main__":
    main()
