"""Cloud-only relative-shape transfer ablation; original final holdout stays closed."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.run_risk import risk_records
from src.data.io import sha256
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import train_origins, evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES
from src.features.relative_transfer import relative_features
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def original_arrays(panel, end, h, stride):
    origins = train_origins(end, h, stride=stride)
    x, scale = relative_features(panel.values[: end + 1], origins, h)
    y = panel.values[origins + h, :, 0].reshape(-1)
    return x, y, scale


def external_arrays(h):
    preparation = json.loads(Path("reports/transfer_preparation.json").read_text())
    source = Path("data/processed/transfer_training.npz")
    if sha256(source) != preparation["processed_sha256"]:
        raise ValueError("External prepared-source checksum mismatch")
    xs, ys, scales = [], [], []
    with np.load(source, allow_pickle=False) as archive:
        for entry in preparation["segments"]:
            values = archive[entry["name"]]
            origins = train_origins(len(values) - 1, h)
            if not len(origins):
                continue
            x, s = relative_features(
                values, origins, h, phase_origins=archive[entry["name"] + "_phase"][origins]
            )
            xs.append(x)
            ys.append(values[origins + h, :, 0].reshape(-1))
            scales.append(s)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(scales)


def fit(x, y, scale, external, ratio, params, rounds, stop=None):
    require_cloud_training()
    valid = np.isfinite(y)
    x, y, scale = x[valid], y[valid], scale[valid]
    weights = scale.copy()
    if ratio > 0:
        ex, ey, es = external
        ev = np.isfinite(ey)
        ex, ey, es = ex[ev], ey[ev], es[ev]
        # Total external influence is controlled independently of target physical units.
        ew = es / es.sum() * weights.sum() * ratio
        x = np.concatenate([x, ex])
        y = np.concatenate([y / scale, ey / es])
        weights = np.concatenate([weights, ew])
    else:
        y = y / scale
    train = lgb.Dataset(x, label=y, weight=weights, feature_name=FEATURE_NAMES)
    kwargs = {}
    if stop is not None:
        sx, sy, ss = stop
        sv = np.isfinite(sy)
        sx, sy, ss = sx[sv], sy[sv], ss[sv]
        validation = lgb.Dataset(sx, label=sy / ss, reference=train)

        def metric(pred, data):
            return "original_mae", float(np.abs(np.maximum(0, pred * ss) - sy).mean()), False

        kwargs = {
            "valid_sets": [validation],
            "feval": metric,
            "callbacks": [lgb.early_stopping(80, verbose=False)],
        }
    return lgb.train(
        {
            **params,
            "metric": "None",
            "verbosity": -1,
            "num_threads": 4,
            "seed": 17,
            "deterministic": True,
            "force_col_wise": True,
        },
        train,
        num_boost_round=rounds,
        **kwargs,
    )


def main():
    require_cloud_training()
    cfg = load_config("configs/transfer_research.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    tiers = assign_load_tiers(panel.values[:504, :, 0])
    rows = []
    records = []
    root = Path("artifacts/transfer_pilot")
    root.mkdir(parents=True, exist_ok=True)
    for h in cfg["horizons"]:
        external = external_arrays(h)
        cx, cy, cs = original_arrays(panel, 419, h, cfg["stride"])
        stop_origins = evaluation_origins(419, 503, h)
        sx, ss = relative_features(panel.values[:504], stop_origins, h)
        sy = panel.values[stop_origins + h, :, 0].reshape(-1)
        fx, fy, fs = original_arrays(panel, 503, h, cfg["stride"])
        origins = evaluation_origins(503, 671, h)
        ax, scale = relative_features(panel.values, origins, h)
        actual = panel.values[origins + h, :, 0]
        for ratio in cfg["external_weight_ratios"]:
            name = f"transfer_{ratio:g}"
            folder = root / name / f"h{h}"
            folder.mkdir(parents=True, exist_ok=True)
            stopping = fit(
                cx, cy, cs, external, ratio, cfg["params"], cfg["max_rounds"], stop=(sx, sy, ss)
            )
            iterations = stopping.best_iteration
            model = fit(fx, fy, fs, external, ratio, cfg["params"], iterations)
            model.save_model(str(folder / "model.txt"))
            loaded = lgb.Booster(model_file=str(folder / "model.txt"))
            predicted = np.maximum(0, loaded.predict(ax, num_threads=4) * scale).reshape(
                actual.shape
            )
            np.testing.assert_allclose(
                predicted.reshape(-1)[:64],
                np.maximum(0, model.predict(ax[:64], num_threads=4) * scale[:64]),
                atol=1e-9,
                rtol=1e-9,
            )
            meta = {
                "schema": cfg["schema"],
                "external_ratio": ratio,
                "fit_end": 503,
                "round_selection_end": 503,
                "iterations": iterations,
                "horizon": h,
                "feature_names": FEATURE_NAMES,
                "research_only": True,
                "holdout_used": False,
                "external_preparation": json.loads(
                    Path("reports/transfer_preparation.json").read_text()
                ),
            }
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            record = ForecastRecord(name, h, origins, panel.beam_ids, actual.copy(), predicted)
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
            pd.DataFrame(rows).to_csv("reports/transfer_pilot_validation.csv", index=False)
            print(
                "Transfer",
                ratio,
                h,
                "h; iterations",
                iterations,
                "MAE",
                float(np.nanmean(np.abs(predicted - actual))),
                flush=True,
            )
    reference = risk_records(Path("reports/forecasts/lightgbm_val.npz"))
    for h in cfg["horizons"]:
        r = reference[h]["lightgbm_c1"]
        record = ForecastRecord(
            "original_c1", h, r["origins"], r["beam_ids"], r["actual"], r["predicted"]
        )
        rows.extend(to_benchmark_rows(record, tiers))
    pd.DataFrame(rows).to_csv("reports/transfer_pilot_validation.csv", index=False)
    save_forecasts(
        records,
        "reports/forecasts/transfer_pilot_val.npz",
        {"research_only": True, "holdout_used": False},
    )
    manifest = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(
        json.dumps({"files": manifest, "schema": cfg["schema"], "holdout_used": False}, indent=2)
    )


if __name__ == "__main__":
    main()
