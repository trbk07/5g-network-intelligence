"""Hosted reproducible multi-corpus rerun with cold initialization evidence."""

from scripts.train_cell_multisource import main


if __name__ == "__main__":
    main("configs/cell_multisource_v2.yaml", "artifacts/cell_multisource_v2", "cell_multisource_v2")
