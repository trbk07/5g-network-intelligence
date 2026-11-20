"""Full-development cloud-only refit, with rounds/epochs fixed before holdout."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.panel import Panel, load_panel
from src.evaluation.origins import train_origins
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.inference.freeze import verify_frozen, config_hash
from src.models.lgbm import FeatureContext, LGBMForecaster, save_lgbm, load_lgbm, SCHEMA
from src.risk.thresholds import fit_thresholds
from src.utils.config import load_config
from src.utils.metadata import collect_run_metadata
from src.utils.runtime import require_cloud_training
from src.data.io import sha256


def main() -> None:
    """Refit selected direct models using development targets only; no validation/holdout Dataset."""
    require_cloud_training()
    frozen, attestation = verify_frozen()
    data = load_config("configs/data.yaml")
    if data["revision"] != frozen["dataset_revision"]:
        raise RuntimeError("Dataset revision changed")
    panel_path = Path(data["processed_dir"]) / "panel.npz"
    if sha256(panel_path) != frozen["panel_sha256"]:
        raise RuntimeError("Canonical cloud panel checksum differs")
    source = load_panel(panel_path)
    end = frozen["final_fit"][1]
    panel = Panel(
        source.values[: end + 1], source.time[: end + 1], source.beam_ids, source.metadata
    )
    directory = Path("artifacts/champion")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "artifacts_manifest.json").exists():
        raise RuntimeError("Final refit already exists; refuse overwrite")
    context = FeatureContext(panel.beam_ids, end)
    if frozen["family"].startswith("lightgbm_"):
        for h in frozen["horizons"]:
            route = frozen["routes"][h]
            origins = train_origins(end, h, stride=frozen["training_stride"])
            x = build_features(panel.values, origins, h, context)
            y = panel.values[origins + h, :, 0].reshape(-1)
            valid = np.isfinite(y)
            params = {
                k: v
                for k, v in route["parameters"].items()
                if k not in ("n_estimators", "early_stopping_rounds")
            }
            params.update(
                metric="mae",
                seed=route["seed"],
                num_threads=frozen["num_threads"],
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            booster = lgb.train(
                params,
                lgb.Dataset(x[valid], label=y[valid], feature_name=FEATURE_NAMES),
                num_boost_round=route["iterations"],
            )
            meta = {
                **collect_run_metadata(data),
                "config": route["parameters"],
                "feature_names": FEATURE_NAMES,
                "feature_schema": SCHEMA,
                "fit_end": end,
                "horizon": h,
                "name": frozen["family"],
                "best_iteration": route["iterations"],
                "num_threads": frozen["num_threads"],
                "seed": route["seed"],
                "train_rows": int(valid.sum()),
                "train_missing_targets": int((~valid).sum()),
                "frozen_config_sha256": config_hash(frozen),
                "freeze_commit": attestation["commit"],
                "early_stopping": False,
                "beam_statistics_in_inputs": False,
            }
            model = LGBMForecaster(booster, context, h, frozen["family"], meta)
            save_lgbm(model, directory / f"h{h}")
            loaded = load_lgbm(directory / f"h{h}")
            np.testing.assert_allclose(
                booster.predict(x[:64], num_threads=4),
                loaded.booster.predict(x[:64], num_threads=4),
                rtol=1e-12,
                atol=1e-12,
            )
            print(
                "Final refit +",
                h,
                "h:",
                route["iterations"],
                "fixed trees; load parity passed",
                flush=True,
            )
    elif frozen["family"].startswith("gru_"):
        raise NotImplementedError(
            "This final refit supports the selected LightGBM champion and baseline fallback only"
        )
    else:
        (directory / "baseline.json").write_text(
            json.dumps({"definition": frozen["family"], "routes": frozen["routes"]}, indent=2)
        )
    thresholds = fit_thresholds(panel, frozen["risk_policy"])
    thresholds.save(directory / "thresholds.npz")
    (directory / "feature_schema.json").write_text(
        json.dumps(
            {
                "family": frozen["family"],
                "features": FEATURE_NAMES if frozen["family"].startswith("lightgbm_") else None,
                "phase_features": 4 if frozen["family"].startswith("gru_") else None,
                "native_missing": frozen["family"].startswith("lightgbm_"),
            },
            indent=2,
        )
    )
    (directory / "champion_meta.json").write_text(
        json.dumps(
            {
                "family": frozen["family"],
                "fit_end": end,
                "dataset_revision": data["revision"],
                "frozen_config_sha256": config_hash(frozen),
                "freeze_commit": attestation["commit"],
                "beam_ids": list(panel.beam_ids),
                "holdout_used_for_fitting": False,
                "threshold_fit_end": end,
            },
            indent=2,
        )
    )
    files = {
        p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()
    }
    (directory / "artifacts_manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "frozen_config_sha256": config_hash(frozen),
                "freeze_commit": attestation["commit"],
            },
            indent=2,
        )
    )
    print("Final champion artifacts ready; no holdout metrics read", flush=True)


if __name__ == "__main__":
    main()
