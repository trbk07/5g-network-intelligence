"""Verify hosted TCN scores and history-only parity; no fitting or selection."""

from scripts.summarize_temporal_linear import summarize
from src.models.temporal_tcn_forecaster import TemporalTCNForecaster


if __name__=="__main__":
    summarize("configs/temporal_tcn.yaml","artifacts/temporal_tcn",TemporalTCNForecaster,
              "temporal_tcn",compare_linear=True,compare_mlp=True)
