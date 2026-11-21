"""Hosted per-source and balanced-pool ablation; no target-network holdout use."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.train_transfer_pilot import external_arrays, fit, original_arrays
from src.data.io import sha256
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.relative_transfer import relative_features
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def balanced_pool(sources):
    """Construct predefined source weights, not learned calibration; equal source mass."""
    arrays = list(sources.values())
    weights = []
    for _, y, scale in arrays:
        valid = np.isfinite(y)
        mass = np.where(valid, scale, 0)
        if mass.sum() <= 0:
            raise ValueError("Source has no observed positive scale mass")
        weights.append(mass / mass.sum() / len(arrays))
    return tuple(np.concatenate([a[i] for a in arrays]) for i in range(3)) + (
        np.concatenate(weights),
    )


def main():
    require_cloud_training()
    overlap = json.loads(Path("reports/transfer_overlap_audit.json").read_text())
    if (
        not overlap["pooled_training_permitted"]
        or sha256(Path("reports/multisource_preparation.json"))
        != overlap["source_preparation_sha256"]
    ):
        raise ValueError("Current multi-source overlap audit required before pooling")
    cfg = load_config("configs/transfer_research.yaml")
    pilot = load_config("configs/multisource_pilot.yaml")
    registry = load_config("configs/transfer_sources.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    root = Path("artifacts/multisource_pilot")
    root.mkdir(parents=True, exist_ok=True)
    tiers = assign_load_tiers(panel.values[:504, :, 0])
    rows = []
    records = []
    for h in cfg["horizons"]:
        sources = {
            s["slug"]: external_arrays(
                h, f"reports/{s['slug']}_preparation.json", cap=pilot["source_row_cap"]
            )
            for s in registry["sources"]
        }
        pool = balanced_pool(sources)
        cx, cy, cs = original_arrays(panel, 419, h, 1)
        so = evaluation_origins(419, 503, h)
        sx, ss = relative_features(panel.values[:504], so, h)
        sy = panel.values[so + h, :, 0].reshape(-1)
        fx, fy, fs = original_arrays(panel, 503, h, 1)
        origins = evaluation_origins(503, 671, h)
        ax, scale = relative_features(panel.values, origins, h)
        actual = panel.values[origins + h, :, 0]
        for case in pilot["source_cases"]:
            external = (
                None
                if case == "target_only"
                else pool
                if case == "balanced_pool"
                else sources[case]
            )
            ratio = 0 if external is None else pilot["external_ratio"]
            stopping = fit(
                cx, cy, cs, external, ratio, cfg["params"], cfg["max_rounds"], stop=(sx, sy, ss)
            )
            model = fit(fx, fy, fs, external, ratio, cfg["params"], stopping.best_iteration)
            directory = root / case / f"h{h}"
            directory.mkdir(parents=True, exist_ok=True)
            model.save_model(str(directory / "model.txt"))
            loaded = lgb.Booster(model_file=str(directory / "model.txt"))
            prediction = np.maximum(0, loaded.predict(ax, num_threads=4) * scale).reshape(
                actual.shape
            )
            np.testing.assert_allclose(
                prediction.reshape(-1)[:64],
                np.maximum(0, model.predict(ax[:64], num_threads=4) * scale[:64]),
                atol=1e-9,
                rtol=1e-9,
            )
            metadata = {
                "schema": cfg["schema"],
                "case": case,
                "horizon": h,
                "rounds": stopping.best_iteration,
                "fit_end": 503,
                "round_selection_end": 503,
                "holdout_used": False,
                "source_row_cap": pilot["source_row_cap"],
                "source_weight_policy": "equal total mass per source in balanced pool; fixed external/original ratio",
                "external_future_metrics_used": False,
                "original_audit": "previously used validation, not fresh test",
            }
            (directory / "meta.json").write_text(json.dumps(metadata, indent=2))
            record = ForecastRecord(case, h, origins, panel.beam_ids, actual.copy(), prediction)
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
            pd.DataFrame(rows).to_csv("reports/multisource_pilot_validation.csv", index=False)
            print(
                "Multi-source",
                case,
                h,
                "h MAE",
                float(np.nanmean(np.abs(prediction - actual))),
                flush=True,
            )
    save_forecasts(
        records,
        "reports/forecasts/multisource_pilot_val.npz",
        {"research_only": True, "holdout_used": False},
    )
    manifest = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(
        json.dumps({"files": manifest, "schema": cfg["schema"]}, indent=2)
    )


if __name__ == "__main__":
    main()
