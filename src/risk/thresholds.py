"""Fit per-beam historical quantiles using only an explicitly supplied fit panel."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.panel import Panel


@dataclass
class Thresholds:
    """Historical P90/P95 plus eligibility guards and observed-only activity."""

    warning: np.ndarray
    critical: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray
    active_ratio: np.ndarray
    n_observed: np.ndarray
    beam_ids: tuple[str, ...]
    fit_end: int

    def save(self, path: str | Path) -> None:
        """Save numerical/string state without pickle."""
        np.savez_compressed(
            path,
            warning=self.warning,
            critical=self.critical,
            valid=self.valid,
            reasons=self.reasons,
            active_ratio=self.active_ratio,
            n_observed=self.n_observed,
            beam_ids=np.array(self.beam_ids),
            fit_end=self.fit_end,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        """Load the same eligibility policy state used by offline risk scoring."""
        with np.load(path, allow_pickle=False) as d:
            return cls(
                d["warning"],
                d["critical"],
                d["valid"],
                d["reasons"],
                d["active_ratio"],
                d["n_observed"],
                tuple(d["beam_ids"].tolist()),
                int(d["fit_end"]),
            )


def fit_thresholds(fit_panel: Panel, cfg: dict) -> Thresholds:
    """Compute historical quantiles and guards; callers must supply no observations after fit cutoff."""
    if (
        cfg["quantiles"] != [0.9, 0.95]
        or cfg["epsilon"] < 0
        or not 0 <= cfg["min_active_ratio"] <= 1
    ):
        raise ValueError("Invalid frozen risk policy")
    target = fit_panel.values[:, :, 0]
    n = np.isfinite(target).sum(0)
    active = np.divide(
        ((target > cfg["epsilon"]) & np.isfinite(target)).sum(0),
        n,
        out=np.zeros(len(n), dtype=float),
        where=n > 0,
    )
    warning, critical = np.full(len(n), np.nan), np.full(len(n), np.nan)
    reasons = np.full(len(n), "eligible", dtype="U32")
    for i in range(len(n)):
        observed = target[np.isfinite(target[:, i]), i]
        if observed.size:
            warning[i], critical[i] = np.quantile(observed, cfg["quantiles"], method="linear")
        if n[i] < cfg["min_observations"]:
            reasons[i] = "insufficient_observations"
        elif warning[i] <= cfg["epsilon"]:
            reasons[i] = "near_zero_threshold"
        elif active[i] < cfg["min_active_ratio"]:
            reasons[i] = "low_activity"
    valid = (reasons == "eligible") & np.isfinite(warning) & np.isfinite(critical)
    return Thresholds(
        warning, critical, valid, reasons, active, n, fit_panel.beam_ids, int(fit_panel.time[-1])
    )
