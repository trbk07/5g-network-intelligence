"""Configuration loading."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML mapping; reject empty or non-mapping configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config
