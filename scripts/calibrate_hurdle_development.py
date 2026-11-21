"""Hosted probability calibration; saved development models stay unchanged."""

import json
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.data.io import sha256
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.features.lightgbm_features import build_features
from src.models.hurdle import mixture_median
from src.models.probability_calibration import calibrated_probability, probability_logits
from src.utils.config import load_config
from src.utils.runtime import require_cloud_training


def main():
    require_cloud_training()
    cfg = load_config("configs/hurdle_calibration.yaml")
    base_cfg = load_config("configs/hurdle_research.yaml")
    root = Path("artifacts/hurdle_calibration")
    root.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    source = Path(cfg["base_checkpoints"])
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(source / name) != digest:
            raise ValueError("Original development checkpoint changed")
    panel, _ = validation_inputs(load_config("configs/data.yaml"))
    values = panel.values[:504]
    rows = []
    for fold in base_cfg["folds"]:
        for h in base_cfg["horizons"]:
            folder = source / fold["name"] / f"h{h}"
            metadata = json.loads((folder / "meta.json").read_text())
            if metadata["protocol"] != base_cfg or metadata["fold"] != fold:
                raise ValueError("Checkpoint development protocol mismatch")
            classifier = lgb.Booster(model_file=str(folder / "active.txt"))
            # Stopping and scoring targets remain disjoint; neither validation nor holdout enters.
            co = evaluation_origins(fold["fit_end"], fold["stop_end"], h)
            so = evaluation_origins(fold["stop_end"], fold["score_end"], h)
            if max(co + h) >= min(so + h) or max(so + h) > 503:
                raise ValueError("Calibration/scoring temporal overlap")
            cx, sx = build_features(values, co, h), build_features(values, so, h)
            cy, sy = values[co + h, :, 0].reshape(-1), values[so + h, :, 0].reshape(-1)
            cp = classifier.predict(cx, num_threads=4)
            sp = classifier.predict(sx, num_threads=4)
            valid = np.isfinite(cy)
            logistic = LogisticRegression(C=cfg["C"], solver=cfg["solver"], max_iter=cfg["max_iter"], tol=cfg["tol"], random_state=cfg["seed"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                logistic.fit(probability_logits(cp[valid], cfg["epsilon"]).reshape(-1, 1), (cy[valid] > 0).astype(int))
            slope, intercept = float(logistic.coef_[0, 0]), float(logistic.intercept_[0])
            calibrated = calibrated_probability(sp, slope, intercept, cfg["epsilon"])
            np.testing.assert_allclose(calibrated, logistic.predict_proba(probability_logits(sp, cfg["epsilon"]).reshape(-1, 1))[:, 1], atol=1e-12, rtol=0)
            conditional = np.column_stack([lgb.Booster(model_file=str(folder / f"q{a:g}.txt")).predict(sx, num_threads=4) for a in base_cfg["conditional_quantiles"]])
            directory = root / fold["name"] / f"h{h}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "calibrator.json").write_text(json.dumps({
                "slope": slope, "intercept": intercept, "epsilon": cfg["epsilon"], "protocol": cfg,
                "fold": fold, "horizon": h, "source_manifest_sha256": sha256(source / "manifest.json"),
                "source_commit": os.environ.get("SOURCE_GIT_COMMIT"), "holdout_used": False,
                "calibration_labels_end": fold["stop_end"], "score_labels_end": fold["score_end"]
            }, indent=2))
            observed = np.isfinite(sy)
            candidates = {}
            for name, probability in (("raw_hurdle", sp), ("calibrated_hurdle", calibrated)):
                prediction = mixture_median(probability, conditional, base_cfg["conditional_quantiles"])
                candidates[name] = prediction
                error = np.abs(prediction[observed] - sy[observed])
                zero = observed & (sy == 0)
                rows.append({"fold": fold["name"], "horizon": h, "model": name,
                             "mae": float(error.mean()), "wape": float(error.sum() / np.abs(sy[observed]).sum()),
                             "zero_fp_rate": float((prediction[zero] > 0).mean()),
                             "brier": brier_score_loss(sy[observed] > 0, probability[observed]),
                             "log_loss": log_loss(sy[observed] > 0, probability[observed], labels=[False, True]),
                             "n_valid": int(observed.sum())})
            with np.load(folder / "score.npz", allow_pickle=False) as previous:
                np.testing.assert_array_equal(previous["actual"], sy)
                np.testing.assert_allclose(previous["hurdle_median"], candidates["raw_hurdle"], atol=1e-10, rtol=0)
            np.savez_compressed(directory / "score.npz", actual=sy, origins=so, beam_ids=np.asarray(panel.beam_ids),
                                raw_probability=sp, calibrated_probability=calibrated, **candidates)
            pd.DataFrame(rows).to_csv("reports/hurdle_calibration_development.csv", index=False)
            print(f"Completed {fold['name']} h{h}: {rows[-2:]}", flush=True)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({"files": files}, indent=2))


if __name__ == "__main__":
    main()
