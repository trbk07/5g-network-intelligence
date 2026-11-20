"""Read-only, checksum-verified historical demo subset; never a live data feed."""

import json
from pathlib import Path

import numpy as np

from src.data.io import sha256
from src.data.panel import load_panel
from src.data.schema import parse_beam_id


class ReplayStore:
    """Load the small replay panel once and expose only copies of causal windows."""

    def __init__(self, directory="demo_assets"):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        for name, digest in self.manifest["files"].items():
            if sha256(self.directory / name) != digest:
                raise ValueError(f"Demo asset checksum mismatch: {name}")
        self.panel = load_panel(self.directory / "replay.npz")
        if not np.array_equal(self.panel.time, np.arange(len(self.panel.time))):
            raise ValueError("Replay time must be contiguous relative hourly indices")
        if list(self.panel.beam_ids) != self.manifest["beam_ids"]:
            raise ValueError("Demo beam axis differs from manifest")
        self.panel.values.flags.writeable = False
        self.panel.time.flags.writeable = False
        self.index = {b: i for i, b in enumerate(self.panel.beam_ids)}
        self.min_origin = int(self.manifest["fit_end"])
        self.max_origin = len(self.panel.time) - 1
        if self.min_origin < 167 or self.min_origin > self.max_origin:
            raise ValueError("Invalid replay range")

    def validate(self, beam_ids, origin):
        """Unknown beams are distinct from invalid replay indices."""
        if (
            not isinstance(origin, int)
            or isinstance(origin, bool)
            or not self.min_origin <= origin <= self.max_origin
        ):
            raise ValueError(f"Origin must lie in {self.min_origin}..{self.max_origin}")
        if not beam_ids or len(beam_ids) > 32 or len(set(beam_ids)) != len(beam_ids):
            raise ValueError("Request 1..32 distinct beams")
        if any(b not in self.index for b in beam_ids):
            raise KeyError("Unknown beam in loaded demo subset")

    def history(self, beam_ids, origin):
        """Return all four observed KPIs for the last 168 hours, in requested beam order."""
        self.validate(beam_ids, origin)
        return self.panel.values[origin - 167 : origin + 1][
            :, [self.index[b] for b in beam_ids]
        ].copy()

    def actual(self, beam_ids, target):
        """Known historical actuals are display-only and never enter a prediction."""
        if target > self.max_origin:
            return np.full(len(beam_ids), np.nan)
        return self.panel.values[target, [self.index[b] for b in beam_ids], 0].copy()

    def meta(self):
        """Describe precisely the loaded subset, not the original full network size."""
        return {
            "beam_ids": list(self.panel.beam_ids),
            "beam_subset_count": len(self.index),
            "origin_min": self.min_origin,
            "origin_max": self.max_origin,
            "lookback": 168,
            "dataset_scope": self.manifest["dataset_scope"],
            "dataset_revision": self.manifest["dataset_revision"],
            "mapping_verified": True,
            "topology": {b: list(parse_beam_id(b)) for b in self.index},
            "is_replay": True,
        }
