"""Shared validation-only data boundary checks for training and replay."""
import json
from pathlib import Path

from src.data.panel import Panel, load_panel
from src.data.splits import build_split_config
from src.utils.config import load_config


def validation_inputs(config: dict) -> tuple[Panel, dict]:
    """Validate frozen split/config and return a panel truncated before holdout."""
    processed = Path(config["processed_dir"])
    panel = load_panel(processed / "panel.npz")
    split = json.loads((processed / "split.json").read_text())
    if split["mode"] not in ("primary", "fallback"):
        raise ValueError("Unknown frozen split mode")
    expected = build_split_config(len(panel.time), split["mode"] == "primary")
    evaluation = load_config(config["evaluation_config"])
    for key in ("fit", "validation", "holdout", "lookback", "horizons", "final_refit"):
        if split[key] != expected[key]:
            raise ValueError(f"Unexpected split boundary: {key}")
        if key != "final_refit" and evaluation[key] != split[key]:
            raise ValueError(f"Evaluation config disagrees: {key}")
    if evaluation["split_mode"] != split["mode"]:
        raise ValueError("Evaluation mode disagrees")
    end = split["validation"][1] + 1
    return Panel(panel.values[:end], panel.time[:end], panel.beam_ids, panel.metadata), expected
