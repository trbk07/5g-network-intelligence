"""Fixed causal seasonal-median reference; no fitted state or tuning."""

import warnings

import numpy as np


class SeasonalMedian:
    """Median of seven observed daily phase matches within 168 hours."""

    name = "seasonal_median_7days"

    def predict(self, history, horizon):
        history = np.asarray(history)
        if (history.ndim != 3 or history.shape[0] != 168 or history.shape[2] != 4
                or not history.shape[1] or np.isinf(history).any()
                or isinstance(horizon, bool) or horizon not in (1, 6, 24)):
            raise ValueError("Expected 168 historical rows and a supported horizon")
        indices = 167+horizon-24*np.arange(1,8)
        # Missing observations stay missing; no interpolation across gaps.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            return np.nanmedian(history[indices,:,0],axis=0)
