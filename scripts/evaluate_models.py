"""Reload LightGBM artifacts and reproduce the validation benchmark without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.io import sha256
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins
from src.evaluation.reports import write_lgbm_analysis
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.models.lgbm import load_lgbm
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata


def main() -> None:
    """Benchmark saved direct models over exactly the baseline validation origin sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lightgbm", "gru"], required=True)
    parser.add_argument("--segment", choices=["val"], default="val")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.diagnostics:
        from scripts.compare_validation import compare

        compare(diagnostics=True)
        return
    if args.model == "gru":
        from scripts.evaluate_gru import evaluate_gru

        evaluate_gru(args.data_config, args.config or "configs/gru.yaml")
        return
    args.config = args.config or "configs/lightgbm.yaml"
    data_config, config = load_config(args.data_config), load_config(args.config)
    panel, split = validation_inputs(data_config)
    reports = Path(data_config["reports_dir"])
    baseline = pd.read_csv(reports / "baseline_benchmark_val.csv")
    baseline_meta = json.loads((reports / "baselines_run_metadata.json").read_text())
    if baseline_meta["dataset_revision"] != data_config["revision"]:
        raise ValueError("Baseline dataset revision differs")
    for key in ("fit", "validation", "holdout", "horizons", "lookback"):
        if baseline_meta["config"]["evaluation"][key] != split[key]:
            raise ValueError(f"Baseline evaluation differs: {key}")
    tiers = assign_load_tiers(panel.values[: split["fit"][1] + 1, :, 0])
    records, rows, importance = [], [], []
    artifact_checksums = {}
    for name in config["configs"]:
        for h in split["horizons"]:
            directory = Path(config["artifact_dir"]) / name / f"h{h}"
            model = load_lgbm(directory)
            if model.static.fit_end != split["fit"][1] or model.static.beam_ids != panel.beam_ids:
                raise ValueError("Artifact fit segment or beam axis mismatch")
            if (
                model.metadata["dataset_revision"] != data_config["revision"]
                or model.metadata["config"] != config["configs"][name]
            ):
                raise ValueError("Artifact dataset or model config differs")
            origins = evaluation_origins(split["fit"][1], split["validation"][1], h)
            features = build_features(panel.values, origins, h, model.static)
            predicted = model.booster.predict(features, num_threads=config["num_threads"])
            predicted = predicted.reshape(len(origins), len(panel.beam_ids))
            # Fast batch path must match independent history-only inference at multiple origins.
            for j in (0, len(origins) // 2, len(origins) - 1):
                t = int(origins[j])
                replay = model.predict_at(panel.values[t - 167 : t + 1], t, h)
                np.testing.assert_allclose(predicted[j], replay, rtol=1e-9, atol=1e-9)
            record = ForecastRecord(
                model.name,
                h,
                origins,
                panel.beam_ids,
                panel.values[origins + h, :, 0].copy(),
                predicted,
            )
            records.append(record)
            benchmark_rows = to_benchmark_rows(record, tiers)
            for row in benchmark_rows:
                row["best_iteration"] = model.metadata["best_iteration"]
                mask = (
                    np.ones(len(tiers), dtype=bool)
                    if row["load_tier"] == "All"
                    else (tiers == row["load_tier"])
                )
                row["negative_predictions"] = int((predicted[:, mask] < 0).sum())
            rows.extend(benchmark_rows)
            for feature, gain, count in zip(
                FEATURE_NAMES,
                model.booster.feature_importance("gain"),
                model.booster.feature_importance("split"),
            ):
                importance.append(
                    {
                        "model": model.name,
                        "horizon": h,
                        "feature": feature,
                        "gain": float(gain),
                        "split_count": int(count),
                    }
                )
            for artifact in ("model.txt", "meta.json", "context.npz"):
                artifact_checksums[str(directory / artifact)] = sha256(directory / artifact)
            print(f"Evaluated {name} +{h}h; batch/history parity passed", flush=True)
    frame, importance_frame = pd.DataFrame(rows), pd.DataFrame(importance)
    frame.to_csv(reports / "lightgbm_benchmark_val.csv", index=False)
    importance_frame.to_csv(reports / "lightgbm_feature_importance.csv", index=False)
    selected, _ = write_lgbm_analysis(records, frame, baseline, importance_frame, reports)
    config["selected_config"] = selected
    Path(args.config).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata = collect_run_metadata({**data_config, "model_config": config, "segment": "val"})
    metadata["artifact_checksums"] = artifact_checksums
    (reports / "lightgbm_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    path = reports / "forecasts/lightgbm_val.npz"
    save_forecasts(records, path, metadata)
    (reports / "lightgbm_forecast_manifest.json").write_text(
        json.dumps(
            {
                "path": "forecasts/lightgbm_val.npz",
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "records": len(records),
                "source_git_commit": metadata["git_commit"],
                "selected_config": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected config by validation: {selected}", flush=True)


if __name__ == "__main__":
    main()
