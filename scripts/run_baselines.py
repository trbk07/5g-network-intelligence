"""Validation-only baseline benchmark: python -m scripts.run_baselines --segment val."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel import load_panel
from src.data.io import sha256
from src.data.splits import build_split_config
from src.evaluation.evaluator import evaluate, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers, mae
from src.evaluation.origins import split_origins
from src.models.baselines import Persistence, SeasonalNaive
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata
from src.utils.seed import set_seed


def main() -> None:
    """Write validation metrics, replay records, load tiers and error analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", choices=["val"], default="val",
                        help="Holdout evaluation is deferred to the frozen Week 5 runner")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    processed, reports = Path(config["processed_dir"]), Path(config["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    panel = load_panel(processed / "panel.npz")
    stored = json.loads((processed / "split.json").read_text())
    split = build_split_config(len(panel.time), stored["mode"] == "primary")
    for key in ("fit", "validation", "holdout", "lookback", "horizons", "final_refit"):
        if stored[key] != split[key]:
            raise ValueError(f"Unexpected frozen split field: {key}")
    evaluation = load_config(config["evaluation_config"])
    if evaluation["split_mode"] != split["mode"]:
        raise ValueError("Evaluation config disagrees with split mode")
    for key in ("fit", "validation", "holdout", "lookback", "horizons"):
        if evaluation[key] != split[key]:
            raise ValueError(f"Evaluation config disagrees with split: {key}")
    (processed / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    # Truncate before evaluation: this runner never supplies holdout observations to a model.
    from src.data.panel import Panel

    end = split["validation"][1] + 1
    panel = Panel(panel.values[:end], panel.time[:end], panel.beam_ids, panel.metadata)
    tiers = assign_load_tiers(panel.values[:split["fit"][1] + 1, :, 0])
    origins = split_origins(split)["validation"]
    records, rows, beam_errors = [], [], []
    for model in (Persistence(), SeasonalNaive(24), SeasonalNaive(168)):
        for horizon in split["horizons"]:
            record = evaluate(model, panel, origins[horizon], horizon, split["lookback"])
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
            for i, beam in enumerate(panel.beam_ids):
                error = mae(record.actual[:, i], record.predicted[:, i])
                beam_errors.append({"model": model.name, "horizon": horizon, "beam_id": beam,
                                    "load_tier": tiers[i], "mae": error.value,
                                    "n_valid": error.n_valid, "n_excluded": error.n_excluded})
            print(f"{model.name} +{horizon}h: {len(record.origins)} origins", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "baseline_benchmark_val.csv", index=False)
    errors = pd.DataFrame(beam_errors)
    errors.to_csv(reports / "baseline_per_beam_val.csv", index=False)
    metadata = collect_run_metadata({**config, "evaluation": evaluation, "segment": args.segment,
                                     "zero_fp_epsilon": 0, "load_tiers_fit_end": split["fit"][1]})
    (reports / "baselines_run_metadata.json").write_text(json.dumps(metadata, indent=2),
                                                        encoding="utf-8")
    forecast_path = reports / "forecasts/baselines_val.npz"
    save_forecasts(records, forecast_path, metadata)
    (reports / "baseline_forecast_manifest.json").write_text(json.dumps({
        "path": "forecasts/baselines_val.npz", "sha256": sha256(forecast_path),
        "size_bytes": forecast_path.stat().st_size, "records": len(records),
        "dataset_revision": config["revision"], "config_sha256": metadata["config_sha256"],
        "source_git_commit": metadata["git_commit"],
    }, indent=2), encoding="utf-8")
    pooled = frame[frame.load_tier == "All"]
    lines = ["# Baseline validation analysis", "", f"Split: {split['mode']}; validation only.",
             "Load tiers use training-only beam means. WAPE values are ratios, not percentages.",
             "Zero-observation FPR means prediction > 0 where actual == 0; it is not risk FPR.",
             "Missing target/prediction exclusions and undefined-metric flags are in the CSV.",
             "Seasonal Naive 24h equals Persistence at horizon +24h; tied winners are reported.",
             "No holdout benchmark was run. No model fitting or tuning was performed.", "",
             "## Best baselines by horizon", ""]
    for horizon, group in pooled.groupby("horizon"):
        best = group.mae.min()
        names = group.loc[np.isclose(group.mae, best, rtol=0, atol=1e-12), "model"].tolist()
        lines.append(f"- +{horizon}h: {', '.join(names)}; MAE={best:.8g}.")
    for title, data in (("Overall, active and zero metrics", pooled),
                        ("Load tier metrics", frame[frame.load_tier != "All"])):
        lines += ["", f"## {title}", "", "```csv", data.to_csv(index=False).strip(), "```"]
    lines += ["", "## Top 10 beam errors per model and horizon", "", "```csv"]
    top = errors.sort_values("mae", ascending=False).groupby(["model", "horizon"]).head(10)
    lines += [top.to_csv(index=False).strip(), "```", ""]
    (reports / "baseline_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
