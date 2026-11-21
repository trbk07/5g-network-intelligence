"""Hosted fixed-round refit; reused-validation audit only, never final holdout."""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.io import sha256
from src.evaluation.evaluator import ForecastRecord, save_forecasts, to_benchmark_rows
from src.evaluation.metrics import assign_load_tiers
from src.evaluation.origins import evaluation_origins, train_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.models.hurdle import mixture_median
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def component_rounds(metadata):
    """Fixed rounded median of development early-stopping rounds; no score tuning."""
    if len(metadata) != 2 or set(metadata[0]["rounds"]) != set(metadata[1]["rounds"]):
        raise ValueError("Two matching development component sets required")
    rounds = {}
    for name in metadata[0]["rounds"]:
        values = [m["rounds"][name] for m in metadata]
        if any(not isinstance(n, int) or n < 1 for n in values):
            raise ValueError("Positive fixed development round counts required")
        rounds[name] = int(np.floor(np.median(values) + 0.5))
    return rounds


def main():
    require_cloud_training()
    cfg = load_config("configs/hurdle_research.yaml")
    source = Path("artifacts/hurdle_research")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(source / name) != digest:
            raise ValueError("Development source checkpoint checksum mismatch")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    root = Path("artifacts/hurdle_candidate")
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    tiers = assign_load_tiers(panel.values[:504, :, 0])
    rows, records = [], []
    for h in cfg["horizons"]:
        metadata = [json.loads((source / fold["name"] / f"h{h}" / "meta.json").read_text()) for fold in cfg["folds"]]
        for meta, fold in zip(metadata, cfg["folds"], strict=True):
            if meta["protocol"] != cfg or meta["fold"] != fold or meta["holdout_used"] or meta["validation_504_671_used"]:
                raise ValueError("Development protocol mismatch")
        rounds = component_rounds(metadata)
        origins = train_origins(503, h, stride=cfg["stride"])
        x = build_features(panel.values[:504], origins, h)
        y = panel.values[origins + h, :, 0].reshape(-1)
        valid = np.isfinite(y)
        directory = root / f"h{h}"
        directory.mkdir(parents=True, exist_ok=True)
        # Only fit labels <=503 enter any component. Audit features are built after fitting.
        models = {}
        for name, count in rounds.items():
            params = {**cfg["params"], "verbosity": -1, "metric": "None",
                      "num_threads": cfg["threads"], "seed": cfg["seed"],
                      "deterministic": True, "force_col_wise": True}
            if name == "active":
                params["objective"] = "binary"
                mask, label = valid, (y[valid] > 0).astype(float)
            elif name == "direct":
                params["objective"] = "regression_l1"
                mask, label = valid, y[valid]
            else:
                alpha = float(name[1:])
                if alpha not in cfg["conditional_quantiles"]:
                    raise ValueError("Unexpected conditional component")
                params.update(objective="quantile", alpha=alpha)
                mask = valid & (y > 0)
                label = y[mask]
            model = lgb.train(params, lgb.Dataset(x[mask], label=label, feature_name=FEATURE_NAMES), num_boost_round=count)
            model.save_model(str(directory / f"{name}.txt"))
            models[name] = lgb.Booster(model_file=str(directory / f"{name}.txt"))
            print(f"Saved h{h} {name} fixed rounds={count}", flush=True)
        audit_origins = evaluation_origins(503, 671, h)
        features = build_features(panel.values, audit_origins, h)
        actual = panel.values[audit_origins + h, :, 0]
        pred = {name: model.predict(features, num_threads=cfg["threads"]) for name, model in models.items()}
        conditional = np.column_stack([pred[f"q{a:g}"] for a in cfg["conditional_quantiles"]])
        candidates = {"hurdle_direct_l1": np.maximum(0, pred["direct"]),
                      "hurdle_median": mixture_median(pred["active"], conditional, cfg["conditional_quantiles"])}
        for name, prediction in candidates.items():
            record = ForecastRecord(name, h, audit_origins, panel.beam_ids, actual.copy(), prediction.reshape(actual.shape))
            records.append(record)
            rows.extend(to_benchmark_rows(record, tiers))
        (directory / "meta.json").write_text(json.dumps({
            "protocol": cfg, "rounds": rounds, "fit_end": 503,
            "round_policy": "rounded_median_of_two_development_component_round_counts",
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "source_commit": os.environ.get("SOURCE_GIT_COMMIT"),
            "horizon": h, "holdout_used": False, "research_only": True,
            "audit": "reused validation 504-671; not a fresh test"
        }, indent=2))
        pd.DataFrame(rows).to_csv("reports/hurdle_candidate_validation.csv", index=False)
        print(pd.DataFrame(rows).query("load_tier == 'All'")[["model", "horizon", "mae", "wape"]].to_string(index=False), flush=True)
    save_forecasts(records, "reports/forecasts/hurdle_candidate_val.npz", {"holdout_used": False, "research_only": True})
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
