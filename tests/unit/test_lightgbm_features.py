import numpy as np
import pytest

from src.data.panel import Panel
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.features.static_stats import compute_static
from tests.utils.perturb import perturb_after


def fixture_panel():
    t = np.arange(240)[:, None, None]
    values = t + np.arange(3)[None, :, None] * 10 + np.arange(4)[None, None, :] * 1000
    return Panel(values.astype(float), np.arange(240), ("0_0_0", "0_0_1", "0_0_2"), {})


def test_44_features_hand_calculated():
    panel = fixture_panel()
    static = compute_static(panel, 199)
    x = build_features(panel.values, np.array([200]), 6, static)
    assert x.shape == (3, 44) and len(set(FEATURE_NAMES)) == 44
    row = dict(zip(FEATURE_NAMES, x[0]))
    expected = {"DLPRB_lag_0": 200, "DLPRB_lag_1": 199, "DLPRB_lag_167": 33,
                "DLPRB_target_seasonal_1d": 182, "DLPRB_target_seasonal_7d": 38,
                "DLPRB_target_seasonal_mean_7d": 110,
                "DLPRB_roll_6_mean": 197.5, "DLPRB_roll_6_std": np.sqrt(35 / 12),
                "DLPRB_roll_24_mean": 188.5, "DLPRB_roll_168_mean": 116.5,
                "DLPRB_roll_168_max": 200, "target_phase_24": 14, "target_phase_168": 38,
                "DLThpVol_lag_0": 1200, "DLThpVol_lag_24": 1176,
                "DLThpVol_roll_6_mean": 1197.5, "beam_fit_mean": 99.5,
                "beam_fit_p90": 179.1, "beam_fit_active_ratio": 199 / 200}
    for name, value in expected.items():
        assert row[name] == pytest.approx(value, rel=1e-6), name


@pytest.mark.parametrize("h", [1, 6, 24])
def test_future_canary_and_batch_history_parity(h):
    panel = fixture_panel()
    panel.values[170:176, 0] = 0
    static = compute_static(panel, 167)
    for origin in (180, 200):
        original = build_features(panel.values, np.array([origin]), h, static)
        changed = build_features(perturb_after(panel, origin).values, np.array([origin]), h, static)
        np.testing.assert_array_equal(original, changed)
        local = build_features(panel.values[origin-167:origin+1], np.array([167]), h, static,
                                phase_origins=np.array([origin]))
        np.testing.assert_array_equal(original, local)
    batch = build_features(panel.values, np.array([180, 200]), h, static)
    assert batch.shape == (6, 44)
    np.testing.assert_array_equal(batch[3:], original)


def test_static_fit_canary_missing_and_zero_policy():
    panel = fixture_panel()
    panel.values[:, 0] = np.nan
    panel.values[:, 1] = 0
    static = compute_static(panel, 167)
    changed = compute_static(perturb_after(panel, 167), 167)
    np.testing.assert_array_equal(static.values, changed.values)
    assert np.isnan(static.values[0]).all()
    np.testing.assert_array_equal(static.values[1], np.zeros(4))
    features = build_features(panel.values, np.array([180]), 1, static)
    assert np.isnan(features[0, FEATURE_NAMES.index("DLPRB_roll_6_mean")])
    assert features[1, FEATURE_NAMES.index("DLPRB_roll_6_std")] == 0
    with pytest.raises(ValueError):
        build_features(panel.values, np.array([166]), 1, static)
    with pytest.raises(ValueError):
        build_features(panel.values, np.array([180]), 25, static)
