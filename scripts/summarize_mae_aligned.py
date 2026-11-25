"""Verify MAE-aligned forecasts and compare unchanged historical control axes."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_neural_hurdle import DirectForecaster, HurdleForecaster
from scripts.summarize_temporal_linear import summarize
from src.models.mae_aligned import MAEAlignedMLP
from src.utils.config import load_config


class ClippedForecaster(DirectForecaster):
    protocol_name = "mae-aligned-multivariate-development-v1"
    architecture_name = "mae-aligned-direct_clipped-v1"
    mode = "direct_clipped"

    def build_model(self, variant, metadata):
        if variant != "B":
            raise ValueError("Four KPI channels required")
        cfg = metadata["protocol"]
        return MAEAlignedMLP(4, cfg["hidden_size"], cfg["dropout"], self.preprocessor.mean[0],
                             self.preprocessor.scale[0], self.mode, cfg["auxiliary_weight"])


class PointHurdleForecaster(ClippedForecaster):
    architecture_name = "mae-aligned-hurdle_mae-v1"
    mode = "hurdle_mae"


def main():
    cfg = load_config("configs/mae_aligned.yaml")
    rows = []
    for mode, adapter, historical, control_adapter in (
            ("direct_clipped", ClippedForecaster, "direct", DirectForecaster),
            ("hurdle_mae", PointHurdleForecaster, "hurdle", HurdleForecaster)):
        summarize("configs/mae_aligned.yaml", f"artifacts/mae_aligned/{mode}", adapter,
                  f"mae_aligned_{mode}", compare_mlp=True)
        for fold in cfg["folds"]:
            control_root = Path("artifacts/neural_hurdle")/historical
            control_adapter(control_root, fold["name"], "B")  # Verify historical manifest/provenance.
            for h in cfg["horizons"]:
                new_path = Path("artifacts/mae_aligned")/mode/fold["name"]/"B"/f"score_h{h}.npz"
                old_path = control_root/fold["name"]/"B"/f"score_h{h}.npz"
                with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
                    for key in ("actual", "origins", "beam_ids"):
                        np.testing.assert_array_equal(new[key], old[key])
                    mask = np.isfinite(new["actual"])
                    if not np.isfinite(new["predicted"][mask]).all() or not np.isfinite(old["predicted"][mask]).all():
                        raise ValueError("Missing predictions cannot shrink the control population")
                    mae = float(np.abs(new["predicted"][mask]-new["actual"][mask]).mean())
                    baseline = float(np.abs(old["predicted"][mask]-old["actual"][mask]).mean())
                    rows.append({"mode": mode, "fold": fold["name"], "horizon": h,
                                 "mae": mae, "control": historical, "control_mae": baseline,
                                 "n_valid": int(mask.sum()),
                                 "relative_mae_gain_vs_control_pct": 100*(1-mae/baseline)})
    frame = pd.DataFrame(rows)
    frame.to_csv("reports/mae_aligned_control_comparison.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
