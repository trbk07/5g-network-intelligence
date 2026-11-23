"""MLP adapter reusing the tested checkpoint/history validation contract."""

from src.models.temporal_linear_forecaster import TemporalLinearForecaster
from src.models.temporal_mlp import TemporalResidualMLP


class TemporalMLPForecaster(TemporalLinearForecaster):
    protocol_name="temporal-residual-mlp-development-v1"
    architecture_name="temporal-residual-mlp-v1"

    def build_model(self,variant,metadata):
        cfg=metadata["protocol"]
        return TemporalResidualMLP(1 if variant=="A" else 4,cfg["hidden_size"],cfg["dropout"])
