"""Nested chronological development experiments. This script never opens final holdout targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.target_recipe import RecipeModel, train_recipe
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def fold_arrays(panel, fold, horizon, stride):
    """Fit targets <=fit_end, stopping targets next, scoring strictly after stop_end."""
    fit_origins = train_origins(fold["fit_end"], horizon, stride=stride)
    stop_origins = evaluation_origins(fold["fit_end"], fold["stop_end"], horizon)
    score_origins = evaluation_origins(fold["stop_end"], fold["score_end"], horizon)
    assert (fit_origins + horizon).max() <= fold["fit_end"]
    assert (stop_origins + horizon).max() < (score_origins + horizon).min()

    def arrays(origins):
        x = build_features(panel.values, origins, horizon)
        return x, panel.values[origins + horizon, :, 0].reshape(-1)

    return (
        fit_origins,
        stop_origins,
        score_origins,
        arrays(fit_origins),
        arrays(stop_origins),
        arrays(score_origins),
    )


def save_score(path, prediction, actual, origins, beam_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predicted=prediction.reshape(len(origins), -1),
        actual=actual.reshape(len(origins), -1),
        origins=origins,
        beam_ids=np.array(beam_ids),
    )


def run_lgbm(panel, cfg):
    """Screen all recipes once, then confirm only predeclared promoted recipes on a later fold."""
    root = Path("artifacts/improvement/lgbm")
    rows = []
    promoted = {}
    for fold_index, fold in enumerate(cfg["folds"]):
        for h in cfg["horizons"]:
            names = list(cfg["recipes"]) if fold_index == 0 else promoted[h]
            cache = {}
            for name in names:
                recipe = cfg["recipes"][name]
                stride = recipe.get("stride", cfg["training_stride"])
                if stride not in cache:
                    cache[stride] = fold_arrays(panel, fold, h, stride)
                origins, _, score_origins, (x, y), (sx, sy), (qx, qy) = cache[stride]
                folder = root / fold["name"] / name / f"h{h}"
                model = train_recipe(
                    x,
                    y,
                    sx,
                    sy,
                    recipe,
                    seed=cfg["seed"],
                    threads=cfg["threads"],
                    max_rounds=cfg["max_rounds"],
                    patience=cfg["patience"],
                    origins=np.repeat(origins, len(panel.beam_ids)),
                    fit_end=fold["fit_end"],
                    metadata={
                        "name": name,
                        "fold": fold,
                        "horizon": h,
                        "protocol": cfg["protocol_version"],
                    },
                )
                model.save(folder)
                loaded = RecipeModel.load(folder)
                prediction = loaded.predict(qx)
                np.testing.assert_allclose(
                    prediction[:64], model.predict(qx[:64]), atol=1e-10, rtol=1e-10
                )
                valid = np.isfinite(qy) & np.isfinite(prediction)
                mae = float(np.abs(prediction[valid] - qy[valid]).mean())
                save_score(folder / "score.npz", prediction, qy, score_origins, panel.beam_ids)
                rows.append(
                    {
                        "family": "lightgbm",
                        "fold": fold["name"],
                        "model": name,
                        "horizon": h,
                        "mae": mae,
                        "best_iteration": model.metadata["best_iteration"],
                        "n_valid": int(valid.sum()),
                        "fit_end": fold["fit_end"],
                        "stop_end": fold["stop_end"],
                        "score_end": fold["score_end"],
                    }
                )
                pd.DataFrame(rows).to_csv("reports/improvement_lgbm_cv.csv", index=False)
                print(
                    "LGBM",
                    fold["name"],
                    name,
                    h,
                    "MAE",
                    mae,
                    "trees",
                    model.metadata["best_iteration"],
                    flush=True,
                )
            if fold_index == 0:
                ranked = sorted(
                    [r for r in rows if r["fold"] == fold["name"] and r["horizon"] == h],
                    key=lambda r: r["mae"],
                )
                promoted[h] = list(
                    dict.fromkeys(
                        ["c1", "deeper"] + [r["model"] for r in ranked[: cfg["promotion_count"]]]
                    )
                )
                Path("reports/improvement_promoted.json").write_text(json.dumps(promoted, indent=2))


def run_catboost(panel, cfg):
    """Complementary GPU family on exactly the same causal40 inputs and chronological folds."""
    from catboost import CatBoostRegressor

    root = Path("artifacts/improvement/catboost")
    rows = []
    for fold in cfg["folds"]:
        for h in cfg["horizons"]:
            _, _, score_origins, (x, y), (sx, sy), (qx, qy) = fold_arrays(
                panel, fold, h, cfg["training_stride"]
            )
            valid, stop_valid = np.isfinite(y), np.isfinite(sy)
            for name, params in cfg["catboost"].items():
                model = CatBoostRegressor(
                    **params,
                    loss_function="MAE",
                    eval_metric="MAE",
                    task_type="GPU",
                    random_seed=cfg["seed"],
                    thread_count=cfg["threads"],
                    allow_writing_files=False,
                    verbose=False,
                )
                model.fit(
                    x[valid],
                    y[valid],
                    eval_set=(sx[stop_valid], sy[stop_valid]),
                    early_stopping_rounds=cfg["patience"],
                    use_best_model=True,
                )
                folder = root / fold["name"] / name / f"h{h}"
                folder.mkdir(parents=True, exist_ok=True)
                model.save_model(str(folder / "model.cbm"))
                loaded = CatBoostRegressor()
                loaded.load_model(str(folder / "model.cbm"))
                prediction = np.maximum(0, loaded.predict(qx))
                np.testing.assert_allclose(
                    prediction[:64], np.maximum(0, model.predict(qx[:64])), atol=1e-9, rtol=1e-9
                )
                valid_score = np.isfinite(qy) & np.isfinite(prediction)
                mae = float(np.abs(prediction[valid_score] - qy[valid_score]).mean())
                metadata = {
                    "family": "catboost",
                    "feature_count": 40,
                    "schema": "causal40",
                    "fold": fold,
                    "horizon": h,
                    "params": params,
                    "seed": cfg["seed"],
                    "best_iteration": model.tree_count_,
                    "research_only": True,
                    "holdout_used": False,
                    "task_type": "GPU",
                }
                (folder / "meta.json").write_text(json.dumps(metadata, indent=2))
                save_score(folder / "score.npz", prediction, qy, score_origins, panel.beam_ids)
                rows.append(
                    {
                        "family": "catboost",
                        "fold": fold["name"],
                        "model": name,
                        "horizon": h,
                        "mae": mae,
                        "best_iteration": model.tree_count_,
                        "n_valid": int(valid_score.sum()),
                    }
                )
                pd.DataFrame(rows).to_csv("reports/improvement_catboost_cv.csv", index=False)
                print(
                    "CatBoost GPU",
                    fold["name"],
                    name,
                    h,
                    "MAE",
                    mae,
                    "trees",
                    model.tree_count_,
                    flush=True,
                )


def main():
    require_cloud_training()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["lgbm", "catboost"], required=True)
    args = parser.parse_args()
    cfg = load_config("configs/model_improvement.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    # This search cannot even index the previously seen outer validation or final holdout.
    from src.data.panel import Panel

    panel = Panel(panel.values[:504], panel.time[:504], panel.beam_ids, panel.metadata)
    if cfg["holdout_allowed"] or max(f["score_end"] for f in cfg["folds"]) > 503:
        raise RuntimeError("Development-only protocol violated")
    Path("reports").mkdir(exist_ok=True)
    (run_lgbm if args.family == "lgbm" else run_catboost)(panel, cfg)
    files = {
        p.as_posix(): sha256(p) for p in Path("artifacts/improvement").rglob("*") if p.is_file()
    }
    Path(f"reports/improvement_{args.family}_manifest.json").write_text(
        json.dumps({"files": files, "holdout_used": False}, indent=2)
    )


if __name__ == "__main__":
    main()
