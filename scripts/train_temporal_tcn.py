"""Hosted-only four-fit causal TCN development study."""

from scripts.train_temporal_mlp import train_study
from src.models.temporal_tcn import TemporalResidualTCN


def build_model(cfg,variant):
    return TemporalResidualTCN(1 if variant=="A" else 4,cfg["channels"],cfg["levels"],cfg["dropout"])


if __name__=="__main__":
    train_study("configs/temporal_tcn.yaml","artifacts/temporal_tcn","temporal_tcn",
                "temporal-residual-tcn-v1",build_model)
