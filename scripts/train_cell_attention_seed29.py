"""Hosted independent-seed robustness check; do not choose the best seed afterward."""

from scripts.train_cell_attention import run_study


if __name__ == "__main__":
    run_study("configs/cell_attention_seed29.yaml", "artifacts/cell_attention_seed29", "cell_attention_seed29")
