"""Reconstruct cloud calibration metrics and predictions without any fitting."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.hurdle import mixture_median
from src.models.probability_calibration import calibrated_probability
from src.utils.config import load_config


def probability_scores(actual, probability):
    actual, probability = np.asarray(actual), np.asarray(probability)
    if (actual.shape != probability.shape or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Aligned finite probabilities required")
    observed = np.isfinite(actual)
    if not observed.any():
        raise ValueError("Observed targets required")
    target = (actual[observed] > 0).astype(float)
    p = probability[observed]
    clip = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {"brier": float(np.mean((p - target) ** 2)),
            "log_loss": float(-np.mean(target * np.log(clip) + (1-target) * np.log1p(-clip)))}


def verify_manifest(root):
    files = json.loads((root / "manifest.json").read_text())["files"]
    for name, digest in files.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return len(files)


def main():
    cfg = load_config("configs/hurdle_calibration.yaml")
    base_cfg = load_config("configs/hurdle_research.yaml")
    root, source = Path("artifacts/hurdle_calibration"), Path(cfg["base_checkpoints"])
    file_count = verify_manifest(root)
    verify_manifest(source)
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    values = panel.values[:504]
    rows, parity = [], []
    for fold in base_cfg["folds"]:
        for h in base_cfg["horizons"]:
            directory, base = root / fold["name"] / f"h{h}", source / fold["name"] / f"h{h}"
            meta = json.loads((directory / "calibrator.json").read_text())
            if (meta["protocol"] != cfg or meta["fold"] != fold or meta["horizon"] != h
                    or meta["holdout_used"] or meta["calibration_labels_end"] != fold["stop_end"]
                    or meta["score_labels_end"] != fold["score_end"] or fold["score_end"] > 503
                    or meta["source_manifest_sha256"] != sha256(source / "manifest.json")):
                raise ValueError("Calibration temporal/source contract mismatch")
            with np.load(directory / "score.npz", allow_pickle=False) as record:
                origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
                np.testing.assert_array_equal(record["origins"], origins)
                np.testing.assert_array_equal(record["beam_ids"], panel.beam_ids)
                actual = values[origins+h, :, 0].reshape(-1)
                np.testing.assert_array_equal(record["actual"], actual)
                features = build_features(values, origins, h)
                raw = lgb.Booster(model_file=str(base / "active.txt")).predict(features, num_threads=4)
                np.testing.assert_allclose(raw, record["raw_probability"], atol=1e-10, rtol=0)
                corrected = calibrated_probability(raw, meta["slope"], meta["intercept"], meta["epsilon"])
                np.testing.assert_allclose(corrected, record["calibrated_probability"], atol=1e-12, rtol=0)
                conditional = np.column_stack([lgb.Booster(model_file=str(base / f"q{a:g}.txt")).predict(features, num_threads=4)
                                               for a in base_cfg["conditional_quantiles"]])
                with np.load(base / "score.npz", allow_pickle=False) as previous:
                    np.testing.assert_array_equal(previous["actual"], actual)
                    np.testing.assert_allclose(previous["hurdle_median"], record["raw_hurdle"], atol=1e-10, rtol=0)
                observed = np.isfinite(actual)
                for name, probability in (("raw_hurdle", raw), ("calibrated_hurdle", corrected)):
                    prediction = mixture_median(probability, conditional, base_cfg["conditional_quantiles"])
                    np.testing.assert_allclose(prediction, record[name], atol=1e-10, rtol=0)
                    parity.append({"fold": fold["name"], "horizon": h, "model": name,
                                   "predictions_checked": len(prediction), "max_error": float(np.max(np.abs(prediction-record[name])))})
                    errors = np.abs(prediction[observed]-actual[observed])
                    zero = observed & (actual == 0)
                    rows.append({"fold": fold["name"], "horizon": h, "model": name,
                                 "mae": float(errors.mean()), "wape": float(errors.sum()/np.abs(actual[observed]).sum()),
                                 "zero_fp_rate": float((prediction[zero] > 0).mean()), "n_valid": int(observed.sum()),
                                 **probability_scores(actual, probability)})
                print(f"Verified {fold['name']} h{h}", flush=True)
    frame = pd.DataFrame(rows)
    saved = pd.read_csv("reports/hurdle_calibration_development.csv")
    columns = ["fold", "horizon", "model", "mae", "wape", "zero_fp_rate", "n_valid", "brier", "log_loss"]
    pd.testing.assert_frame_equal(frame[columns], saved[columns], check_dtype=False, atol=1e-12, rtol=1e-12)
    comparison = frame.pivot(index=["fold", "horizon"], columns="model", values="mae")
    comparison["relative_mae_gain_pct"] = 100*(1-comparison.calibrated_hurdle/comparison.raw_hurdle)
    comparison.to_csv("reports/hurdle_calibration_comparison.csv")
    pd.DataFrame(parity).to_csv("reports/hurdle_calibration_parity.csv", index=False)
    Path("reports/hurdle_calibration_verification.json").write_text(json.dumps({
        "artifact_files_verified": file_count, "predictions_recomputed": sum(p["predictions_checked"] for p in parity),
        "max_prediction_error": max(p["max_error"] for p in parity), "all_metrics_reconstructed": True,
        "validation_504_671_used": False, "holdout_used": False,
        "claim_scope": "reused development folds only; no generalization or production-promotion claim"
    }, indent=2))
    print(comparison.to_string())


if __name__ == "__main__":
    main()
