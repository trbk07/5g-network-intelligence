"""Select ensemble only on inner chronological scores; fixed refit, then one reused-validation audit."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.io import sha256
from src.data.panel import Panel
from src.data.schema import parse_beam_id
from src.evaluation.bootstrap import paired_cluster_ci
from src.evaluation.common_mask import comparable_mask
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.research_ensemble import ResearchEnsemble, baseline_from_features
from src.models.target_recipe import RecipeModel, train_recipe
from src.risk.evaluator import evaluate_exceedance
from src.risk.thresholds import fit_thresholds
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def load_inner(panel, cfg, h):
    """Intersect actually confirmed recipes; align original observations and beam axes exactly."""
    by_fold = []
    for fold in cfg["folds"]:
        group = {}
        actual = None
        for family in ("lgbm", "catboost"):
            for path in Path("artifacts/improvement", family, fold["name"]).glob(
                f"*/h{h}/score.npz"
            ):
                with np.load(path, allow_pickle=False) as data:
                    if actual is None:
                        actual = data["actual"].copy()
                        origins = data["origins"].copy()
                    else:
                        np.testing.assert_array_equal(actual, data["actual"])
                        np.testing.assert_array_equal(origins, data["origins"])
                    if tuple(data["beam_ids"].tolist()) != panel.beam_ids:
                        raise ValueError("Inner score beam ordering differs")
                    group[path.parent.parent.name] = data["predicted"].reshape(-1).copy()
        if actual is None:
            raise RuntimeError("No completed inner predictions")
        x = build_features(panel.values, origins, h)
        for name in ("persistence", "seasonal24"):
            group[name] = baseline_from_features(x, name)
        by_fold.append((actual.reshape(-1), group))
    available = set.intersection(*(set(g) for _, g in by_fold))
    y = np.concatenate([a for a, _ in by_fold])
    predictions = {n: np.concatenate([g[n] for _, g in by_fold]) for n in available}
    mask = np.isfinite(y)
    for prediction in predictions.values():
        mask &= np.isfinite(prediction)
    return y[mask], {n: p[mask] for n, p in predictions.items()}


def choose_weights(actual, predictions, cfg):
    """Cloud-only grid calibration; not permitted as local 'statistics' masquerading as training."""
    require_cloud_training()
    scores = {n: float(np.abs(actual - p).mean()) for n, p in predictions.items()}
    ranked = sorted(scores, key=lambda n: (scores[n], n))
    pool = ranked[:4]
    # Include a complementary family and an actual baseline even when their individual MAE is higher.
    for candidates in (
        [n for n in ranked if n.startswith("cb")],
        ["c1"],
        [n for n in ranked if n in ("persistence", "seasonal24")],
    ):
        if candidates and candidates[0] not in pool:
            pool.append(candidates[0])
    denominator = round(1 / cfg["ensemble"]["weight_step"])
    best = {"mae": float("inf"), "weights": {}, "bias": 0.0}
    for size in range(1, min(cfg["ensemble"]["max_members"], len(pool)) + 1):
        for names in itertools.combinations(pool, size):
            for counts in itertools.product(range(1, denominator + 1), repeat=size):
                if sum(counts) != denominator:
                    continue
                weights = np.asarray(counts) / denominator
                prediction = sum(w * predictions[n] for w, n in zip(weights, names))
                mae = float(np.abs(prediction - actual).mean())
                if mae < best["mae"] - 1e-9:
                    best = {"mae": mae, "weights": dict(zip(names, weights.tolist())), "bias": 0.0}
    # Bias is calibrated after blending; it uses neither outer validation nor holdout.
    blend = sum(w * predictions[n] for n, w in best["weights"].items())
    for bias in cfg["ensemble"]["bias_grid"]:
        mae = float(np.abs(np.maximum(0, blend + bias) - actual).mean())
        if mae < best["mae"] - 1e-9:
            best.update(mae=mae, bias=bias)
    best.update(
        single_model_mae=scores, shortlist=pool, selection_data="inner1+inner2 score windows only"
    )
    return best


def refit_member(panel, cfg, h, name, folder):
    """Pick rounds with stopping <=503, then refit <=503 without any audit-set early stopping."""
    if name in ("persistence", "seasonal24"):
        return {"family": "baseline", "path": None, "iterations": None}
    family = "catboost" if name in cfg["catboost"] else "lgbm"
    stride = (
        cfg["recipes"][name].get("stride", cfg["training_stride"])
        if family == "lgbm"
        else cfg["training_stride"]
    )
    calibration_origins = train_origins(419, h, stride=stride)
    stopping_origins = evaluation_origins(419, 503, h)
    final_origins = train_origins(503, h, stride=stride)

    def arrays(origins):
        return build_features(panel.values, origins, h), panel.values[origins + h, :, 0].reshape(-1)

    x, y = arrays(calibration_origins)
    sx, sy = arrays(stopping_origins)
    fx, fy = arrays(final_origins)
    folder.mkdir(parents=True, exist_ok=True)
    if family == "lgbm":
        recipe = cfg["recipes"][name]
        stopping = train_recipe(
            x,
            y,
            sx,
            sy,
            recipe,
            seed=cfg["seed"],
            threads=cfg["threads"],
            max_rounds=cfg["max_rounds"],
            patience=cfg["patience"],
            origins=np.repeat(calibration_origins, len(panel.beam_ids)),
            fit_end=419,
        )
        iterations = stopping.metadata["best_iteration"]
        stopping.save(folder / "stopping")
        model = train_recipe(
            fx,
            fy,
            None,
            None,
            recipe,
            seed=cfg["seed"],
            threads=cfg["threads"],
            fixed_rounds=iterations,
            origins=np.repeat(final_origins, len(panel.beam_ids)),
            fit_end=503,
            metadata={"fit_end": 503, "horizon": h, "name": name, "early_stopping_on_audit": False},
        )
        model.save(folder)
        loaded = RecipeModel.load(folder)
        np.testing.assert_allclose(
            model.predict(fx[:64]), loaded.predict(fx[:64]), atol=1e-10, rtol=1e-10
        )
    else:
        from catboost import CatBoostRegressor

        params = {
            **cfg["catboost"][name],
            "task_type": "GPU",
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": cfg["seed"],
            "thread_count": cfg["threads"],
            "verbose": False,
            "allow_writing_files": False,
        }
        valid, sv, fv = np.isfinite(y), np.isfinite(sy), np.isfinite(fy)
        stopping = CatBoostRegressor(**params)
        stopping.fit(
            x[valid],
            y[valid],
            eval_set=(sx[sv], sy[sv]),
            early_stopping_rounds=cfg["patience"],
            use_best_model=True,
        )
        iterations = stopping.tree_count_
        stopping.save_model(str(folder / "stopping.cbm"))
        params["iterations"] = iterations
        model = CatBoostRegressor(**params)
        model.fit(fx[fv], fy[fv])
        model.save_model(str(folder / "model.cbm"))
        loaded = CatBoostRegressor()
        loaded.load_model(str(folder / "model.cbm"))
        np.testing.assert_allclose(
            model.predict(fx[:64]), loaded.predict(fx[:64]), atol=1e-9, rtol=1e-9
        )
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "family": "catboost",
                    "fit_end": 503,
                    "horizon": h,
                    "iterations": iterations,
                    "feature_count": 40,
                    "research_only": True,
                    "early_stopping_on_audit": False,
                },
                indent=2,
            )
        )
    print("Fixed development refit", name, h, "rounds", iterations, flush=True)
    return {
        "family": family,
        "path": folder.relative_to(Path("artifacts/improved_candidate")).as_posix(),
        "iterations": iterations,
    }


def main():
    require_cloud_training()
    cfg = load_config("configs/model_improvement.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    development = Panel(panel.values[:504], panel.time[:504], panel.beam_ids, panel.metadata)
    output = Path("artifacts/improved_candidate")
    output.mkdir(parents=True, exist_ok=True)
    choices = {}
    for h in cfg["horizons"]:
        y, predictions = load_inner(development, cfg, h)
        choices[h] = choose_weights(y, predictions, cfg)
        print("Inner-only selection", h, choices[h], flush=True)
    # Commit the weight/recipe selection before the audit arrays or metrics are accessed.
    Path("reports/improvement_selection.json").write_text(json.dumps(choices, indent=2))
    selection_sha = sha256(Path("reports/improvement_selection.json"))
    plan = {
        "schema": "causal40-ensemble-research-v1",
        "fit_end": 503,
        "research_only": True,
        "deployment_allowed": False,
        "holdout_used": False,
        "audit_is_reused_validation": True,
        "selection_sha256": selection_sha,
        "routes": {},
    }
    for h in cfg["horizons"]:
        route = {"bias": choices[h]["bias"], "members": {}}
        for name, weight in choices[h]["weights"].items():
            member = refit_member(development, cfg, h, name, output / "members" / name / f"h{h}")
            route["members"][name] = {**member, "weight": weight}
        plan["routes"][str(h)] = route
    (output / "ensemble.json").write_text(json.dumps(plan, indent=2))
    model = ResearchEnsemble(output)
    tiers = assign_load_tiers(development.values[:, :, 0])
    thresholds = fit_thresholds(development, load_config("configs/risk.yaml"))
    old = risk_records(Path("reports/forecasts/lightgbm_val.npz"))
    week7 = risk_records(Path("reports/forecasts/week7_research_val.npz"))
    clusters = np.array([parse_beam_id(b)[0] for b in panel.beam_ids])
    records, rows, paired, risks = [], [], [], []
    for h in cfg["horizons"]:
        origins = evaluation_origins(503, 671, h)
        x = build_features(panel.values, origins, h)
        actual = panel.values[origins + h, :, 0]
        prediction = model.predict_features(x, h).reshape(actual.shape)
        for j in (0, len(origins) // 2, len(origins) - 1):
            t = int(origins[j])
            np.testing.assert_allclose(
                prediction[j],
                model.predict_at(panel.values[t - 167 : t + 1], t, h),
                atol=1e-9,
                rtol=1e-9,
            )
        group = {
            "improved_ensemble": prediction,
            "lightgbm_c1": old[h]["lightgbm_c1"]["predicted"],
            "week7_deeper_l1": week7[h]["deeper_l1"]["predicted"],
        }
        np.testing.assert_array_equal(actual, old[h]["lightgbm_c1"]["actual"])
        np.testing.assert_array_equal(actual, week7[h]["deeper_l1"]["actual"])
        mask = comparable_mask(actual, group)
        for name, p in group.items():
            record = ForecastRecord(
                name,
                h,
                origins,
                panel.beam_ids,
                np.where(mask, actual, np.nan),
                np.where(mask, p, np.nan),
            )
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
            risks.extend(
                {"model": name, "horizon": h, **r}
                for r in evaluate_exceedance(actual, p, thresholds, mask)
            )
            if name != "improved_ensemble":
                paired.append(
                    {
                        "horizon": h,
                        "reference": name,
                        **paired_cluster_ci(actual, prediction, p, clusters, mask),
                    }
                )
        print(
            "Reused-validation audit",
            h,
            "MAE",
            float(np.abs(prediction[mask] - actual[mask]).mean()),
            flush=True,
        )
    pd.DataFrame(rows).to_csv("reports/improvement_audit.csv", index=False)
    pd.DataFrame(paired).to_csv("reports/improvement_paired.csv", index=False)
    pd.DataFrame(risks).to_csv("reports/improvement_risk.csv", index=False)
    save_forecasts(
        records,
        "reports/forecasts/improvement_audit.npz",
        {"research_only": True, "selection_sha256": selection_sha, "holdout_used": False},
    )
    if sha256(Path("reports/improvement_selection.json")) != selection_sha:
        raise RuntimeError("Selection changed after audit")
    files = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*") if p.is_file()}
    (output / "manifest.json").write_text(
        json.dumps(
            {"files": files, "selection_sha256": selection_sha, "holdout_used": False}, indent=2
        )
    )


if __name__ == "__main__":
    main()
