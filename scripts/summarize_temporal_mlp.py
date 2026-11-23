"""Verify MLP checkpoints, full scoring coverage and history-only CPU/GPU parity."""

from scripts.summarize_temporal_linear import summarize
from src.models.temporal_mlp_forecaster import TemporalMLPForecaster


if __name__=="__main__":
    summarize("configs/temporal_mlp.yaml","artifacts/temporal_mlp",TemporalMLPForecaster,"temporal_mlp",compare_linear=True)
