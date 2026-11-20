"""Causal target transform tests; only the marked round-trip test trains on hosted cloud."""

import numpy as np
import pytest

from src.features.lightgbm_features import FEATURE_NAMES, build_features
from src.models.target_recipe import RecipeModel, components, decode, train_recipe


def test_normalization_restores_original_unit_l1():
    x = np.zeros((3, 40), dtype=np.float32)
    x[:, FEATURE_NAMES.index("DLPRB_roll_168_mean")] = [0, 2, 4]
    x[:, FEATURE_NAMES.index("DLPRB_target_seasonal_mean_7d")] = [1, 3, 5]
    anchor, scale = components(x, "normalized_residual")
    y = np.array([0.5, 4, 6])
    raw = np.array([0, 0.3, 1])
    encoded = (y - anchor) / scale
    np.testing.assert_allclose(np.abs(raw - encoded) * scale, np.abs(anchor + scale * raw - y))
    np.testing.assert_allclose(decode(raw, x, "normalized_residual"), anchor + scale * raw)
    assert scale[0] == 0.1


def test_recipe_features_ignore_future_and_never_add_inputs():
    values = np.random.default_rng(17).random((220, 2, 4))
    before = build_features(values, np.array([180]), 6)
    values[181:] = 1e9
    after = build_features(values, np.array([180]), 6)
    np.testing.assert_array_equal(before, after)
    for recipe in (
        "direct",
        "normalized",
        "residual_last",
        "residual_seasonal",
        "normalized_residual",
    ):
        np.testing.assert_array_equal(
            decode(np.ones(2), before, recipe), decode(np.ones(2), after, recipe)
        )
    assert before.shape == (2, 40)


def test_cloud_guard_precedes_any_fit():
    from src.utils.runtime import is_cloud_runtime

    if is_cloud_runtime():
        pytest.skip("Guard denial is a local-only test")
    with pytest.raises(RuntimeError, match="Local model training is forbidden"):
        train_recipe(None, None, None, None, {"transform": "direct", "params": {}})


@pytest.mark.cloud_training
def test_cloud_recipe_save_load_and_fixed_rounds(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.random((150, 40), dtype=np.float32)
    y = x[:, 0] * 2
    recipe = {
        "transform": "normalized_residual",
        "params": {"num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 5},
    }
    model = train_recipe(x[:100], y[:100], x[100:], y[100:], recipe, max_rounds=15, patience=3)
    model.save(tmp_path / "model")
    np.testing.assert_allclose(model.predict(x), RecipeModel.load(tmp_path / "model").predict(x))
    fixed = train_recipe(x, y, None, None, recipe, fixed_rounds=3)
    assert fixed.booster.current_iteration() == 3
