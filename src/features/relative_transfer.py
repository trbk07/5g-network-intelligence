"""Research features invariant to target-unit scaling, using observed history only."""

import numpy as np

from src.features.lightgbm_features import FEATURE_NAMES, build_features


def relative_features(values, origins, horizon, scale_floor=1e-6, phase_origins=None):
    x = build_features(values, origins, horizon, phase_origins=phase_origins)
    scale = x[:, FEATURE_NAMES.index("DLPRB_roll_168_mean")].astype(float)
    scale = np.maximum(np.nan_to_num(scale, nan=0), scale_floor)
    result = x.copy()
    result[:, :23] /= scale[:, None]
    # Deliberate target-only ablation: no incompatible cross-domain proxy KPIs.
    result[:, 25:] = np.nan
    return result, scale
