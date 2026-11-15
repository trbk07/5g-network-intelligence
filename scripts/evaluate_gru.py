"""Validation-only GRU replay, seed summaries and A/B ablation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.models.gru_forecaster import GRUForecaster
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata


def evaluate_gru(data_config_path: str = "configs/data.yaml", config_path: str = "configs/gru.yaml") -> None:
    """Reload every declared run and compare full-beam validation with LightGBM/baselines."""
    data_config, config = load_config(data_config_path), load_config(config_path)
    panel, split = validation_inputs(data_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    if device == "cpu" and len(panel.beam_ids) > 32:
        raise RuntimeError("Full GRU replay requires GPU; CPU replay reserved for small tests")
    reports = Path(data_config["reports_dir"])
    tiers = assign_load_tiers(panel.values[:split["fit"][1]+1, :, 0])
    origin_sets = {h: evaluation_origins(split["fit"][1], split["validation"][1], h)
                   for h in (1, 6, 24)}
    records, rows, failures = [], [], []
    for variant in config["variants"]:
        for seed in config["seeds"]:
            directory = Path(config["artifact_dir"]) / variant / f"seed{seed}"
            model = GRUForecaster(directory, device)
            if (model.metadata["dataset_revision"] != data_config["revision"]
                    or model.preprocessor.fit_end != split["fit"][1]):
                raise ValueError("GRU artifact revision/fit boundary mismatch")
            all_predictions = model.predict_origins(panel, origin_sets[1], config["batch_size"])
            t = int(origin_sets[1][0])
            np.testing.assert_allclose(all_predictions[0], model.predict_all(panel.values[t-167:t+1], t),
                                       rtol=1e-4, atol=1e-5)
            for k, h in enumerate((1, 6, 24)):
                origins = origin_sets[h]
                record = ForecastRecord(model.name, h, origins, panel.beam_ids,
                    panel.values[origins+h, :, 0].copy(), all_predictions[:len(origins), :, k])
                records.append(record)
                current = to_benchmark_rows(record, tiers)
                for row in current:
                    row.update(variant=variant, seed=seed, best_epoch=model.metadata["best_epoch"])
                rows.extend(current)
                absolute = np.abs(record.actual-record.predicted)
                valid = np.flatnonzero(np.isfinite(absolute))
                for flat in valid[np.argsort(absolute.reshape(-1)[valid])[-3:][::-1]]:
                    oi, bi = np.unravel_index(flat, absolute.shape)
                    failures.append({"model": model.name, "horizon": h,
                        "origin": int(origins[oi]), "beam_id": panel.beam_ids[bi],
                        "actual": record.actual[oi, bi], "prediction": record.predicted[oi, bi],
                        "absolute_error": absolute[oi, bi]})
            print(f"GRU {variant} seed{seed}: full-beam validation replay passed", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(reports/"gru_benchmark_val.csv", index=False)
    pooled = frame[frame.load_tier == "All"]
    summary = pooled.groupby(["variant", "horizon"]).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"), seeds=("seed", "nunique"),
        active_only_mae_mean=("active_only_mae", "mean"),
        zero_fp_rate_mean=("zero_fp_rate", "mean")).reset_index()
    summary.to_csv(reports/"ablation_study.csv", index=False)
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(reports/"gru_failure_examples.csv", index=False)
    baseline = pd.read_csv(reports/"baseline_benchmark_val.csv")
    best = baseline[baseline.load_tier == "All"].groupby("horizon").mae.min()
    summary["best_baseline_mae"] = summary.horizon.map(best)
    summary["skill_vs_baseline"] = 1-summary.mae_mean/summary.best_baseline_mae
    lightgbm = pd.read_csv(reports/"lightgbm_skill_val.csv")
    selected = load_config("configs/lightgbm.yaml")["selected_config"]
    lgb = lightgbm[lightgbm.model == f"lightgbm_{selected}"].set_index("horizon").mae
    summary["lightgbm_mae"] = summary.horizon.map(lgb)
    summary["skill_vs_lightgbm"] = 1-summary.mae_mean/summary.lightgbm_mae
    summary.to_csv(reports/"gru_comparison_val.csv", index=False)
    metadata = collect_run_metadata({**data_config, "gru": config, "segment": "val"})
    (reports/"gru_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    save_forecasts(records, reports/"forecasts/gru_val.npz", metadata)
    lines = ["# GRU validation error analysis", "", "No holdout evaluated. Predictions are un-clipped.",
             "Mean +/- sample SD across seeds is descriptive only; two seeds do not establish stability.",
             "One-seed SD is undefined. Early stopping uses joint-horizon validation origins; final replay",
             "uses the same per-horizon full-beam origin sets as LightGBM and baselines.", ""]
    for title, table in (("A/B and model comparison", summary), ("Load/active/zero metrics", frame),
                          ("Largest validation errors (not incident labels)", failure_frame)):
        lines += [f"## {title}", "", "```csv", table.to_csv(index=False).strip(), "```", ""]
    (reports/"gru_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
