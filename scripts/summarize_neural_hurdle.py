"""Verify both hosted neural heads against saved forecasts and historical controls."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_temporal_linear import summarize
from src.models.temporal_mlp_forecaster import TemporalMLPForecaster
from src.models.neural_hurdle import NeuralHurdleMLP


class DirectForecaster(TemporalMLPForecaster):
    protocol_name = "multivariate-neural-hurdle-development-v1"
    architecture_name = "multivariate-neural-direct-v1"
    kind = "direct"

    def __init__(self, root, fold_name, variant):
        super().__init__(root, fold_name, variant)
        np.testing.assert_equal(float(self.model.target_mean), self.preprocessor.mean[0])
        np.testing.assert_equal(float(self.model.target_scale), self.preprocessor.scale[0])
        np.testing.assert_equal(float(self.model.zero_scaled),
                                float(np.float32(-self.preprocessor.mean[0]/self.preprocessor.scale[0])))

    def build_model(self, variant, metadata):
        if variant != "B":
            raise ValueError("Multivariate neural study requires variant B")
        cfg = metadata["protocol"]
        return NeuralHurdleMLP(4, cfg["hidden_size"], cfg["dropout"],
                               self.preprocessor.mean[0], self.preprocessor.scale[0], self.kind)


class HurdleForecaster(DirectForecaster):
    architecture_name = "multivariate-neural-hurdle-v1"
    kind = "hurdle"


def main():
    for kind, adapter in (("direct", DirectForecaster), ("hurdle", HurdleForecaster)):
        summarize("configs/neural_hurdle.yaml", f"artifacts/neural_hurdle/{kind}", adapter,
                  f"neural_hurdle_{kind}", compare_mlp=True)
    keys = ["fold", "variant", "horizon"]
    direct = pd.read_csv("reports/neural_hurdle_direct_development.csv").set_index(keys).sort_index()
    hurdle = pd.read_csv("reports/neural_hurdle_hurdle_development.csv").set_index(keys).sort_index()
    if not direct.index.equals(hurdle.index):
        raise ValueError("Direct/hurdle scoring coverage differs")
    np.testing.assert_array_equal(direct.n_valid, hurdle.n_valid)
    comparison = hurdle.copy()
    comparison["direct_mae"] = direct.mae
    comparison["relative_mae_gain_vs_direct_pct"] = 100*(1-hurdle.mae/direct.mae)
    comparison["direct_zero_fp_rate"] = direct.zero_fp_rate
    comparison.to_csv(Path("reports/neural_hurdle_matched_comparison.csv"))
    print(comparison.to_string())


if __name__ == "__main__":
    main()
