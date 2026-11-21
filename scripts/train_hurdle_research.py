"""Hosted-only development comparison of direct L1 and zero-inflated medians."""

import json
import os
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.optimize_models import fold_arrays
from src.data.io import sha256
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES
from src.models.hurdle import mixture_median
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def fit_component(x, y, sx, sy, cfg, objective, alpha=None):
    require_cloud_training()
    params = {**cfg["params"], "objective": objective, "verbosity": -1,
              "num_threads": cfg["threads"], "seed": cfg["seed"],
              "deterministic": True, "force_col_wise": True}
    if alpha is not None:
        params["alpha"] = alpha
    params["metric"] = {"binary": "binary_logloss", "quantile": "quantile",
                        "regression_l1": "l1"}[objective]
    mask, smask = np.isfinite(y), np.isfinite(sy)
    if not mask.any() or not smask.any():
        raise ValueError("Empty component fit/stopping population")
    return lgb.train(params, lgb.Dataset(x[mask], label=y[mask], feature_name=FEATURE_NAMES),
                     num_boost_round=cfg["max_rounds"],
                     valid_sets=[lgb.Dataset(sx[smask], label=sy[smask])],
                     callbacks=[lgb.early_stopping(cfg["patience"], verbose=False)])


def main():
    require_cloud_training()
    cfg = load_config("configs/hurdle_research.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    # The experiment never indexes targets from reused validation or final holdout.
    panel = replace(panel, values=panel.values[:504], time=panel.time[:504])
    root = Path("artifacts/hurdle_research")
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    rows = []
    for fold in cfg["folds"]:
        if not fold["fit_end"] < fold["stop_end"] < fold["score_end"] <= 503:
            raise ValueError("Development chronology must end by 503")
        for h in cfg["horizons"]:
            _, _, origins, (x, y), (sx, sy), (qx, qy) = fold_arrays(panel, fold, h, cfg["stride"])
            directory = root / fold["name"] / f"h{h}"
            directory.mkdir(parents=True, exist_ok=True)
            models = {}
            class_y = np.where(np.isfinite(y), y > 0, np.nan)
            class_sy = np.where(np.isfinite(sy), sy > 0, np.nan)
            models["active"] = fit_component(x, class_y, sx, class_sy, cfg, "binary")
            models["direct"] = fit_component(x, y, sx, sy, cfg, "regression_l1")
            yp, syp = np.where(y > 0, y, np.nan), np.where(sy > 0, sy, np.nan)
            for alpha in cfg["conditional_quantiles"]:
                models[f"q{alpha:g}"] = fit_component(x, yp, sx, syp, cfg, "quantile", alpha)
            predictions = {}
            for name, model in models.items():
                path = directory / f"{name}.txt"
                model.save_model(str(path))
                loaded = lgb.Booster(model_file=str(path))
                predictions[name] = loaded.predict(qx, num_threads=cfg["threads"])
                np.testing.assert_allclose(predictions[name][:64], model.predict(qx[:64]), atol=1e-10)
            positive = np.column_stack([predictions[f"q{a:g}"] for a in cfg["conditional_quantiles"]])
            candidates = {
                "direct_l1": np.maximum(0, predictions["direct"]),
                "hurdle_median": mixture_median(predictions["active"], positive, cfg["conditional_quantiles"]),
            }
            valid = np.isfinite(qy)
            for name, prediction in candidates.items():
                if not np.isfinite(prediction[valid]).all():
                    raise ValueError("Nonfinite scored prediction")
                errors = np.abs(prediction[valid] - qy[valid])
                zero = valid & (qy == 0)
                rows.append({"fold": fold["name"], "horizon": h, "model": name,
                             "mae": float(errors.mean()),
                             "wape": float(errors.sum() / np.abs(qy[valid]).sum()),
                             "zero_fp_rate": float((prediction[zero] > 0).mean()),
                             "n_valid": int(valid.sum())})
            np.savez_compressed(directory / "score.npz", actual=qy, origins=origins,
                                beam_ids=np.asarray(panel.beam_ids), **candidates)
            metadata = {"protocol": cfg, "fold": fold, "horizon": h,
                        "rounds": {name: m.best_iteration for name, m in models.items()},
                        "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
                        "holdout_used": False, "validation_504_671_used": False}
            (directory / "meta.json").write_text(json.dumps(metadata, indent=2))
            pd.DataFrame(rows).to_csv("reports/hurdle_development_scores.csv", index=False)
            print(f"Completed {fold['name']} h{h}: {rows[-2:]}", flush=True)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
