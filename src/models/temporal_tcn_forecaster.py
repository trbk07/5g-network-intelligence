"""Research-only TCN loader sharing causal history and checkpoint provenance checks."""

from src.models.temporal_linear_forecaster import TemporalLinearForecaster
from src.models.temporal_tcn import TemporalResidualTCN


class TemporalTCNForecaster(TemporalLinearForecaster):
    protocol_name="temporal-residual-tcn-development-v1"
    architecture_name="temporal-residual-tcn-v1"

    def build_model(self,variant,metadata):
        cfg=metadata["protocol"]
        return TemporalResidualTCN(1 if variant=="A" else 4,cfg["channels"],cfg["levels"],cfg["dropout"])
