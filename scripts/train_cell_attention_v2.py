"""Hosted seeded/strict-backend rerun; original v1 artifacts remain immutable."""

from scripts.train_cell_attention import run_study


if __name__ == "__main__":
    run_study("configs/cell_attention_v2.yaml", "artifacts/cell_attention_v2", "cell_attention_v2")
