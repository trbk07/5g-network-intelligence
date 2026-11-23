"""Reconstruct neural development scores and history-only CPU/GPU parity, without fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_hurdle_research import score_rows
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.models.temporal_linear_forecaster import TemporalLinearForecaster
from src.utils.config import load_config


def summarize(config_path="configs/temporal_linear.yaml",artifact_path="artifacts/temporal_linear",
              forecaster_class=TemporalLinearForecaster,report_prefix="temporal_linear",compare_linear=False):
    """Shared neural replay verification; architecture-specific contracts stay in loaders."""
    cfg = load_config(config_path)
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    root = Path(artifact_path)
    rows, comparisons, parity = [], [], []
    for fold in cfg["folds"]:
        for variant in cfg["variants"]:
            model = forecaster_class(root, fold["name"], variant)
            if model.metadata["protocol"] != cfg:
                raise ValueError("Downloaded protocol differs")
            directory = root / fold["name"] / variant
            for h in cfg["horizons"]:
                with np.load(directory / f"score_h{h}.npz", allow_pickle=False) as score:
                    origins = evaluation_origins(fold["stop_end"], fold["score_end"], h)
                    actual = panel.values[origins+h, :, 0].reshape(-1)
                    np.testing.assert_array_equal(score["origins"], origins)
                    np.testing.assert_array_equal(score["actual"], actual)
                    np.testing.assert_array_equal(score["beam_ids"], panel.beam_ids)
                    metrics = score_rows(actual, {variant: score["predicted"]}, fold["name"], h)[0]
                    rows.append({"fold": fold["name"], "variant": variant, "horizon": h,
                                 **{key: metrics[key] for key in ("mae", "wape", "zero_fp_rate", "n_valid")}})
                    reference_path = Path("artifacts/hurdle_research") / fold["name"] / f"h{h}" / "score.npz"
                    with np.load(reference_path, allow_pickle=False) as reference:
                        np.testing.assert_array_equal(reference["actual"], actual)
                        np.testing.assert_array_equal(reference["origins"], origins)
                        np.testing.assert_array_equal(reference["beam_ids"], panel.beam_ids)
                        mask = np.isfinite(actual)
                        baseline = float(np.abs(reference["hurdle_median"][mask]-actual[mask]).mean())
                    comparisons.append({**rows[-1], "hurdle_mae": baseline,
                                        "relative_mae_gain_vs_hurdle_pct": 100*(1-metrics["mae"]/baseline)})
                    if compare_linear:
                        linear_path=Path("artifacts/temporal_linear")/fold["name"]/variant/f"score_h{h}.npz"
                        with np.load(linear_path,allow_pickle=False) as linear:
                            for key in ("actual","origins","beam_ids"):
                                np.testing.assert_array_equal(linear[key],score[key])
                            observed=np.isfinite(actual)
                            linear_mae=float(np.abs(linear["predicted"][observed]-actual[observed]).mean())
                        comparisons[-1].update(linear_mae=linear_mae,relative_mae_gain_vs_linear_pct=100*(1-metrics["mae"]/linear_mae))
                    beams = np.array([0, len(panel.beam_ids)//2, len(panel.beam_ids)-1])
                    for index in (0, len(origins)-1):
                        origin = int(origins[index])
                        prediction = model.predict_at(panel.values[origin-167:origin+1, beams, :], origin, h)
                        saved = score["predicted"].reshape(len(origins), len(panel.beam_ids))[index, beams]
                        np.testing.assert_allclose(prediction, saved, atol=1e-5, rtol=0)
                        parity.append({"fold": fold["name"], "variant": variant, "horizon": h, "origin": origin,
                                       "beams_checked": len(beams), "max_error": float(np.max(np.abs(prediction-saved)))})
    frame = pd.DataFrame(rows)
    saved = pd.read_csv(f"reports/{report_prefix}_development.csv")
    columns = ["fold", "variant", "horizon", "mae", "wape", "zero_fp_rate", "n_valid"]
    pd.testing.assert_frame_equal(frame[columns], saved[columns], check_dtype=False, atol=1e-12, rtol=1e-12)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(f"reports/{report_prefix}_comparison.csv", index=False)
    pd.DataFrame(parity).to_csv(f"reports/{report_prefix}_cpu_parity.csv", index=False)
    Path(f"reports/{report_prefix}_verification.json").write_text(json.dumps({
        "all_metrics_reconstructed": True, "preprocessing_fit_prefix_verified": True,
        "history_only_beam_predictions_checked": sum(p["beams_checked"] for p in parity),
        "cpu_gpu_tolerance_original_units": 1e-5, "max_parity_error": max(p["max_error"] for p in parity),
        "validation_504_671_used": False, "holdout_used": False, "production_promoted": False,
        "claim_scope": "one-seed reused development study only; no unseen-test generalization claim"
    }, indent=2))
    print(comparison.to_string(index=False))


def main():
    summarize()


if __name__ == "__main__":
    main()
