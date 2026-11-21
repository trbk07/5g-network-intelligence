"""History-only and unit-scaling invariance tests; no fitting."""

import numpy as np

from src.features.relative_transfer import relative_features


def test_units_and_future_do_not_change_relative_shape():
    rng = np.random.default_rng(17)
    values = rng.uniform(0.2, 3, (210, 2, 4))
    a, scale = relative_features(values, np.array([180]), 6)
    changed = values.copy()
    changed[:, :, 0] *= 100
    b, bscale = relative_features(changed, np.array([180]), 6)
    np.testing.assert_allclose(a[:, :25], b[:, :25], atol=2e-6)
    np.testing.assert_allclose(scale * 100, bscale, rtol=1e-6)
    assert np.isnan(a[:, 25:]).all()
    changed = values.copy()
    changed[181:] = 99999
    c, cscale = relative_features(changed, np.array([180]), 6)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(scale, cscale)


def test_fixed_source_balancing_does_not_follow_dataset_size():
    from scripts.train_multisource_pilot import balanced_pool

    small = (np.zeros((2, 40)), np.ones(2), np.array([1.0, 2.0]))
    large = (np.zeros((20, 40)), np.ones(20), np.arange(1, 21, dtype=float))
    _, _, _, weights = balanced_pool({"small": small, "large": large})
    assert np.isclose(weights[:2].sum(), 0.5)
    assert np.isclose(weights[2:].sum(), 0.5)
