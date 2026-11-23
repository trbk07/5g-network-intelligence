"""Research-only fixed MLP blend loader; production routing stays frozen."""

from src.models.blend_research_forecaster import BlendResearchForecaster
from src.models.temporal_mlp import TemporalResidualMLP


class MLPBlendResearchForecaster(BlendResearchForecaster):
    component = "mlp"

    def build_model(self, variant, metadata):
        if (variant not in self.protocol["expected_selected_variants"]
                or metadata["identity"]["epochs"] != self.protocol["expected_fixed_epochs"][variant]
                or metadata["development_manifest_sha256"] != self.protocol["mlp_manifest_sha256"]):
            raise ValueError("MLP refit differs from pinned development/epoch contract")
        cfg=metadata["base_protocol"]
        return TemporalResidualMLP(1 if variant=="A" else 4,cfg["hidden_size"],cfg["dropout"])
