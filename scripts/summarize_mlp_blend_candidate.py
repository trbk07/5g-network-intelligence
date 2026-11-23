"""Fixed MLP candidate inference and validation audit, without fitting."""

from scripts.summarize_blend_candidate import summarize
from src.models.mlp_blend_research_forecaster import MLPBlendResearchForecaster


if __name__ == "__main__":
    summarize("configs/mlp_blend_candidate_audit.yaml","configs/temporal_mlp.yaml",
              "artifacts/mlp_blend_candidate","artifacts/temporal_mlp","mlp_blend_candidate",
              "mlp",MLPBlendResearchForecaster)
