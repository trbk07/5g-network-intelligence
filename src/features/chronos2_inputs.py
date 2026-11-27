"""Causal four-KPI tasks for native multivariate foundation-model fine-tuning."""

import numpy as np


def native_tasks(channel_first):
    """One DLPRB target plus three past-only covariates; no auxiliary future labels."""
    values = np.asarray(channel_first)
    if values.ndim != 3 or values.shape[1] != 4 or not values.shape[0] or not values.shape[2]:
        raise ValueError("Expected beam-by-four-KPI-by-time array")
    return [{"target": item[0], "past_covariates": {
        name: item[index] for index, name in enumerate(("DLThpVol", "DLThpTime", "MR_number"), start=1)}}
        for item in values]


def training_prefix(values, fit_end):
    """Return (beam, KPI, time) without exposing any post-fit label or covariate."""
    values = np.asarray(values)
    if (values.ndim != 3 or values.shape[2] != 4 or not values.shape[1]
            or isinstance(fit_end, bool) or not isinstance(fit_end, (int, np.integer))
            or not 191 <= fit_end <= 503 or fit_end >= len(values)):
        raise ValueError("Four-KPI development prefix with 168+24 hours required")
    prefix = values[:fit_end+1]
    if np.isinf(prefix).any():
        raise ValueError("Infinite fit-prefix values are invalid; NaN remains native missing")
    return prefix.transpose(1, 2, 0).astype(np.float32, copy=True)


def history_tasks(values, origin, beams):
    """Exactly t-167..t for each beam; never read future actual covariates."""
    values, beams = np.asarray(values), np.asarray(beams)
    if (values.ndim != 3 or values.shape[2] != 4 or beams.ndim != 1 or not beams.size
            or not np.issubdtype(beams.dtype, np.integer)
            or (beams < 0).any() or (beams >= values.shape[1]).any()
            or isinstance(origin, bool) or not isinstance(origin, (int, np.integer))
            or not 167 <= origin <= 502 or origin >= len(values)):
        raise ValueError("Valid development origin and original beam indices required")
    history = values[origin-167:origin+1, beams, :]
    if np.isinf(history).any():
        raise ValueError("Infinite history values are invalid")
    return history.transpose(1, 2, 0).astype(np.float32, copy=True)
