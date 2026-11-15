"""Validation error analysis and transparent config selection."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluator import ForecastRecord
from src.evaluation.metrics import mae


def write_lgbm_analysis(records: list[ForecastRecord], benchmark: pd.DataFrame,
                         baseline: pd.DataFrame, importance: pd.DataFrame,
                         reports: Path) -> tuple[str, pd.DataFrame]:
    """Write skill, per-beam errors, >=3 failure cases and importance; select by validation."""
    best_baseline = baseline[baseline.load_tier == "All"].groupby("horizon").mae.min()
    pooled = benchmark[benchmark.load_tier == "All"].copy()
    pooled["best_baseline_mae"] = pooled.horizon.map(best_baseline)
    if pooled.best_baseline_mae.isna().any() or (pooled.best_baseline_mae <= 0).any():
        raise ValueError("Normalized config selection requires positive baseline MAE per horizon")
    pooled["normalized_mae"] = pooled.mae / pooled.best_baseline_mae
    pooled["skill_score"] = 1 - pooled.normalized_mae
    scores = pooled.groupby("model").normalized_mae.mean().sort_values(kind="stable")
    selected = scores.index[0].removeprefix("lightgbm_")
    pooled.to_csv(reports / "lightgbm_skill_val.csv", index=False)
    errors, failures = [], []
    for record in records:
        for i, beam in enumerate(record.beam_ids):
            result = mae(record.actual[:, i], record.predicted[:, i])
            errors.append({"model": record.model, "horizon": record.horizon,
                           "beam_id": beam, "mae": result.value, "n_valid": result.n_valid})
        absolute = np.abs(record.actual - record.predicted)
        valid = np.flatnonzero(np.isfinite(absolute))
        worst = valid[np.argsort(absolute.reshape(-1)[valid])[-3:][::-1]]
        for flat in worst:
            origin_i, beam_i = np.unravel_index(flat, absolute.shape)
            failures.append({"model": record.model, "horizon": record.horizon,
                             "origin": int(record.origins[origin_i]),
                             "target_index": int(record.origins[origin_i] + record.horizon),
                             "beam_id": record.beam_ids[beam_i],
                             "actual": record.actual[origin_i, beam_i],
                             "prediction": record.predicted[origin_i, beam_i],
                             "absolute_error": absolute[origin_i, beam_i]})
    per_beam = pd.DataFrame(errors)
    per_beam.to_csv(reports / "lightgbm_per_beam_val.csv", index=False)
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(reports / "lightgbm_failure_examples.csv", index=False)
    lines = ["# LightGBM validation analysis", "", f"Selected config: **{selected}**.",
             "Selection uses the mean of horizon MAE / best baseline MAE on validation only.",
             "Ties use stable config order. This is not final champion selection or holdout evaluation.",
             "C1 uses regression_l1; C2 uses regression. Both early-stop on validation MAE.",
             "Predictions are un-clipped; negative predictions are retained and reported.",
             "WAPE is a ratio; zero-observation FPR is prediction > 0 on actual == 0.", ""]
    for title, frame in (("Overall skill versus best baseline", pooled),
                          ("MAE/load tiers and active/zero metrics", benchmark),
                          ("Failure examples (largest errors, descriptive only)", failure_frame),
                          ("Top beam errors", per_beam.sort_values("mae", ascending=False).groupby(
                              ["model", "horizon"]).head(10)),
                          ("Top features by gain (not causal attribution)", importance.sort_values(
                              "gain", ascending=False).groupby(["model", "horizon"]).head(10))):
        lines += [f"## {title}", "", "```csv", frame.to_csv(index=False).strip(), "```", ""]
    lines += ["## Sanity checks", ""]
    for _, row in pooled[pooled.horizon == 1].iterrows():
        lines.append(f"- {row.model} +1h MAE / best baseline = {row.normalized_mae:.6f}.")
    lines += ["- Large errors are retrospective examples, not verified congestion events.",
              "- Only the two planned configs were run; no additional hyperparameter search.",
              "- Fit-only static statistics and feature causality are covered by perturbation tests.",
              "- The validation runner truncates all observations before holdout.", ""]
    (reports / "lightgbm_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return selected, pooled
