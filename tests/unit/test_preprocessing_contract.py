import numpy as np
import pytest

from src.data.panel import Panel
from src.features.gru_inputs import GRUPreprocessor, fit_imputer, fit_scaler
from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.features.static_stats import compute_static
from tests.utils.perturb import perturb_after


@pytest.mark.parametrize("variant,k", [("A", 1), ("B", 4)])
def test_global_parameters_cutoff_inverse_and_phase_only(variant, k, tmp_path):
    values = np.random.default_rng(17).normal(size=(220, 9, 4))
    values[10, 1] = np.nan
    panel = Panel(values, np.arange(220), tuple(f"0_0_{i}" for i in range(9)), {})
    prep = GRUPreprocessor.fit(panel, 167, variant)
    changed = GRUPreprocessor.fit(perturb_after(panel, 167), 167, variant)
    assert fit_imputer(panel, 167, variant).shape == (k,)
    assert fit_scaler(panel, 167, prep.median)[0].shape == (k,)
    for key in ("median", "mean", "scale"):
        assert getattr(prep, key).shape == (k,)
        np.testing.assert_array_equal(getattr(prep, key), getattr(changed, key))
    a = GRUPreprocessor.fit(panel, 167, "A")
    b = GRUPreprocessor.fit(panel, 167, "B")
    assert (a.median[0], a.mean[0], a.scale[0]) == (b.median[0], b.mean[0], b.scale[0])
    raw = values[100, :, 0]
    scaled = (raw - prep.mean[0]) / prep.scale[0]
    np.testing.assert_allclose(scaled * prep.scale[0] + prep.mean[0], raw)
    phase = prep.static_at(np.array([180, 180]), np.array([0, 8]))
    assert phase.shape == (2, 4)
    np.testing.assert_array_equal(phase[0], phase[1])
    prep.save(tmp_path / "prep.npz")
    loaded = GRUPreprocessor.load(tmp_path / "prep.npz")
    np.testing.assert_array_equal(
        prep.transform(values, variant), loaded.transform(values, variant)
    )
    with np.load(tmp_path / "prep.npz", allow_pickle=False) as data:
        assert "static" not in data


def test_fit_statistics_cannot_enter_lightgbm_features():
    panel = Panel(
        np.random.default_rng(17).normal(size=(220, 2, 4)), np.arange(220), ("0_0_0", "0_0_1"), {}
    )
    stats = compute_static(panel, 167)
    original = build_features(panel.values, np.array([180]), 6, stats)
    stats.values[:] = 999999
    np.testing.assert_array_equal(original, build_features(panel.values, np.array([180]), 6, stats))
    assert original.shape == (2, 40) and not any("beam_fit" in name for name in FEATURE_NAMES)
