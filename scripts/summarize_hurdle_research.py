"""Verify hosted hurdle checkpoints and saved development scores without fitting."""

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
from src.utils.config import load_config


def score_rows(actual, predictions, fold, horizon):
    """Use exactly the observed target population; missing predictions are errors."""
    actual = np.asarray(actual)
    valid = np.isfinite(actual)
    denominator = np.abs(actual[valid]).sum()
    if not valid.any() or denominator <= 0:
        raise ValueError("Nonempty positive-denominator scoring population required")
    rows = []
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.shape != actual.shape or not np.isfinite(prediction[valid]).all():
            raise ValueError("Every observed target needs a finite aligned prediction")
        error = np.abs(prediction[valid] - actual[valid])
        zero = valid & (actual == 0)
        rows.append({"fold": fold, "horizon": horizon, "model": name,
                     "mae": float(error.mean()), "wape": float(error.sum() / denominator),
                     "n_valid": int(valid.sum()),
                     "zero_fp_rate": float((prediction[zero] > 0).mean()) if zero.any() else None})
    return rows


def main():
    root = Path("artifacts/hurdle_research")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    cfg = load_config("configs/hurdle_research.yaml")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    rows, parity = [], []
    for fold in cfg["folds"]:
        for h in cfg["horizons"]:
            directory = root / fold["name"] / f"h{h}"
            meta = json.loads((directory / "meta.json").read_text())
            if (meta["fold"] != fold or meta["protocol"] != cfg or meta["holdout_used"]
                    or meta["validation_504_671_used"]):
                raise ValueError("Downloaded checkpoint temporal contract differs")
            with np.load(directory / "score.npz", allow_pickle=False) as score:
                origins = score["origins"]
                np.testing.assert_array_equal(origins, evaluation_origins(fold["stop_end"], fold["score_end"], h))
                if not ((origins + h > fold["stop_end"]).all()
                        and (origins + h <= fold["score_end"]).all()):
                    raise ValueError("Scoring origins violate temporal protocol")
                np.testing.assert_array_equal(score["beam_ids"], panel.beam_ids)
                actual = panel.values[origins + h, :, 0].reshape(-1)
                np.testing.assert_array_equal(score["actual"], actual)
                rows.extend(score_rows(actual, {name: score[name] for name in ("direct_l1", "hurdle_median")}, fold["name"], h))
                # Three history-only beams at two score origins per fold/horizon.
                beams = np.array([0, len(panel.beam_ids) // 2, len(panel.beam_ids) - 1])
                models = {name: lgb.Booster(model_file=str(directory / f"{name}.txt")) for name in meta["rounds"]}
                for index in (0, len(origins) - 1):
                    origin = int(origins[index])
                    history = panel.values[origin - 167:origin + 1, beams, :]
                    features = build_features(history, np.array([167]), h, phase_origins=np.array([origin]))
                    pred = {name: m.predict(features, num_threads=4) for name, m in models.items()}
                    conditional = np.column_stack([pred[f"q{a:g}"] for a in cfg["conditional_quantiles"]])
                    candidates = {"direct_l1": np.maximum(0, pred["direct"]),
                                  "hurdle_median": mixture_median(pred["active"], conditional, cfg["conditional_quantiles"])}
                    for name, prediction in candidates.items():
                        saved = score[name].reshape(len(origins), len(panel.beam_ids))[index, beams]
                        np.testing.assert_allclose(prediction, saved, atol=1e-8, rtol=0)
                        parity.append({"fold": fold["name"], "horizon": h, "origin": origin,
                                       "model": name, "max_abs_error": float(np.max(np.abs(prediction - saved)))})
    table = pd.DataFrame(rows)
    saved = pd.read_csv("reports/hurdle_development_scores.csv")
    columns = ["fold", "horizon", "model", "mae", "wape", "n_valid", "zero_fp_rate"]
    pd.testing.assert_frame_equal(table[columns], saved[columns], check_dtype=False, atol=1e-12, rtol=1e-12)
    comparison = table.pivot(index=["fold", "horizon"], columns="model", values="mae")
    comparison["relative_mae_gain_pct"] = 100 * (1 - comparison.hurdle_median / comparison.direct_l1)
    comparison.to_csv("reports/hurdle_development_comparison.csv")
    pd.DataFrame(parity).to_csv("reports/hurdle_cpu_parity.csv", index=False)
    Path("reports/hurdle_verification.json").write_text(json.dumps({
        "artifact_files_verified": len(manifest["files"]), "history_only_beam_predictions_checked": len(parity) * 3,
        "maximum_parity_error": max(r["max_abs_error"] for r in parity),
        "reused_validation_used": False, "holdout_used": False,
        "claim_scope": "exploratory development folds only, no generalization or promotion claim"
    }, indent=2))
    print(comparison.to_string())


if __name__ == "__main__":
    main()
