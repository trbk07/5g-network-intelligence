"""Beam statistics fitted on a declared training prefix only."""
from dataclasses import dataclass

import numpy as np

from src.data.panel import Panel


@dataclass
class StaticStats:
    """Per-beam mean, population std, P90 and observed-only active ratio."""

    values: np.ndarray
    beam_ids: tuple[str, ...]
    fit_end: int


def compute_static(panel: Panel, fit_end: int) -> StaticStats:
    """Compute statistics from indices 0..fit_end; all-missing beams stay NaN."""
    if not isinstance(fit_end, int) or not 0 <= fit_end < len(panel.time):
        raise ValueError("fit_end must identify a training prefix")
    target = panel.values[:fit_end + 1, :, 0]
    result = np.full((len(panel.beam_ids), 4), np.nan)
    for i, column in enumerate(target.T):
        observed = column[np.isfinite(column)]
        if observed.size:
            result[i] = [observed.mean(), observed.std(ddof=0), np.quantile(observed, .9),
                         np.mean(observed > 0)]
    return StaticStats(result, panel.beam_ids, fit_end)
