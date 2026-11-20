"""Post-freeze validation research, isolated from production champion and holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.audit import check_mapping
from src.data.panel import Panel
from src.data.schema import parse_beam_id
from src.evaluation.bootstrap import paired_cluster_ci
from src.evaluation.common_mask import comparable_mask
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.lgbm import FeatureContext, load_lgbm, save_lgbm, train_lgbm
from src.risk.evaluator import evaluate_exceedance
from src.risk.thresholds import fit_thresholds
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    """Use only the original fit/validation ranges; require fresh data before any promotion."""
    require_cloud_training()
    cfg = load_config("configs/research_week7.yaml")
    data = load_config("configs/data.yaml")
    panel, split = validation_inputs(data)
    assert len(panel.time) == split["validation"][1] + 1
    context = FeatureContext(panel.beam_ids, split["fit"][1])
    tiers = assign_load_tiers(panel.values[: context.fit_end + 1, :, 0])
    fit_panel = Panel(
        panel.values[: context.fit_end + 1],
        panel.time[: context.fit_end + 1],
        panel.beam_ids,
        panel.metadata,
    )
    risk_cfg = load_config("configs/risk.yaml")
    thresholds = fit_thresholds(fit_panel, risk_cfg)
    if not check_mapping(panel)["complete_expected_mapping"]:
        raise RuntimeError("Expected verified BS mapping")
    clusters = np.array([parse_beam_id(b)[0] for b in panel.beam_ids])
    reference = risk_records(Path("reports/forecasts/lightgbm_val.npz"))
    records, rows, risks, paired = [], [], [], []
    for h in (1, 6, 24):
        origins = train_origins(context.fit_end, h, stride=cfg["training_stride"])
        val_origins = evaluation_origins(context.fit_end, split["validation"][1], h)
        train_x = build_features(fit_panel.values, origins, h, context)
        train_y = fit_panel.values[origins + h, :, 0].reshape(-1)
        val_x = build_features(panel.values, val_origins, h, context)
        actual = panel.values[val_origins + h, :, 0]
        old = reference[h]["lightgbm_c1"]
        np.testing.assert_array_equal(actual, old["actual"])
        np.testing.assert_array_equal(val_origins, old["origins"])
        predictions = {"lightgbm_c1": old["predicted"]}
        models = {}
        for name, parameters in cfg["configs"].items():
            model = train_lgbm(
                train_x,
                train_y,
                val_x,
                actual.reshape(-1),
                parameters,
                context,
                h,
                name,
                cfg["seed"],
                cfg["num_threads"],
                {
                    "dataset_revision": data["revision"],
                    "research_only": True,
                    "holdout_used": False,
                },
            )
            directory = Path(cfg["artifact_dir"]) / name / f"h{h}"
            save_lgbm(model, directory)
            loaded = load_lgbm(directory)
            prediction = loaded.booster.predict(val_x, num_threads=cfg["num_threads"]).reshape(
                actual.shape
            )
            np.testing.assert_allclose(
                prediction.reshape(-1)[:100],
                model.booster.predict(val_x[:100], num_threads=4),
                rtol=1e-12,
                atol=1e-12,
            )
            predictions[name] = prediction
            models[name] = loaded
            print(
                "Validation research:",
                name,
                h,
                "h; trees:",
                model.metadata["best_iteration"],
                flush=True,
            )
        mask = comparable_mask(actual, predictions)
        for name, prediction in predictions.items():
            record = ForecastRecord(
                name,
                h,
                val_origins,
                panel.beam_ids,
                np.where(mask, actual, np.nan),
                np.where(mask, prediction, np.nan),
            )
            records.append(record)
            for row in to_benchmark_rows(record, tiers):
                row["best_iteration"] = (
                    models[name].metadata["best_iteration"] if name in models else None
                )
                rows.append(row)
            risks.extend(
                {"model": name, "horizon": h, **r}
                for r in evaluate_exceedance(
                    actual, prediction, thresholds, mask, risk_cfg["min_positive_report"]
                )
            )
            if name != "lightgbm_c1":
                paired.append(
                    {
                        "model": name,
                        "horizon": h,
                        **paired_cluster_ci(
                            actual, prediction, predictions["lightgbm_c1"], clusters, mask
                        ),
                    }
                )
    reports = Path("reports")
    pd.DataFrame(rows).to_csv(reports / "week7_research_validation.csv", index=False)
    pd.DataFrame(risks).to_csv(reports / "week7_research_risk.csv", index=False)
    pd.DataFrame(paired).to_csv(reports / "week7_research_paired.csv", index=False)
    save_forecasts(
        records,
        reports / "forecasts/week7_research_val.npz",
        {"research_only": True, "holdout_used": False},
    )
    frame = pd.DataFrame(rows)
    totals = frame[frame.load_tier == "All"].pivot(index="model", columns="horizon", values="mae")
    scores = (totals / totals.loc["lightgbm_c1"]).mean(axis=1)
    best = str(scores.idxmin())
    decision = {
        "research_only": True,
        "production_champion_unchanged": True,
        "holdout_reopened": False,
        "validation_best": best,
        "mean_normalized_mae": scores.to_dict(),
        "limitation": "Repeated validation experiments are exploratory. Fresh unseen chronological data required for a new final claim or promotion.",
    }
    (reports / "week7_research_decision.json").write_text(json.dumps(decision, indent=2))
    (reports / "week7_research.md").write_text(
        "# Week 7 validation research\n\n"
        "Four predeclared parameter sets use the original 40 causal inputs. All training runs on hosted Kaggle CPU.\n"
        "The squared-error candidate tests whether stronger peak sensitivity helps historical exceedance recall at a cost to MAE.\n"
        "No threshold tuning, holdout reopening, production replacement or new final generalization claim.\n\n"
        + totals.to_csv()
        + "\n\n"
        + json.dumps(decision, indent=2)
    )


if __name__ == "__main__":
    main()
