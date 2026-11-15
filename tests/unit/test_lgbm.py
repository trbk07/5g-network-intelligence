import numpy as np
import pytest

from src.data.panel import Panel
from src.evaluation.evaluator import evaluate
from src.features.lightgbm_features import build_features
from src.features.static_stats import compute_static
from src.models.lgbm import load_lgbm, save_lgbm, train_lgbm
from tests.utils.perturb import assert_causal

pytestmark = pytest.mark.cloud_training


def test_training_missing_targets_determinism_save_load_and_model_canary(tmp_path):
    rng = np.random.default_rng(17)
    values = rng.uniform(size=(240, 3, 4))
    values[180, 0, 0] = np.nan
    panel = Panel(values, np.arange(240), ("0_0_0", "0_0_1", "0_0_2"), {})
    static = compute_static(panel, 199)
    train_o, val_o = np.arange(167, 199), np.arange(199, 225)
    x = build_features(values, train_o, 1, static)
    v = build_features(values, val_o, 1, static)
    y, vy = values[train_o + 1, :, 0].reshape(-1), values[val_o + 1, :, 0].reshape(-1)
    config = {"objective": "regression_l1", "num_leaves": 4, "min_data_in_leaf": 3,
              "learning_rate": .1, "n_estimators": 20, "early_stopping_rounds": 3}
    model = train_lgbm(x, y, v, vy, config, static, 1, "tiny", num_threads=1)
    assert model.metadata["train_missing_targets"] == 1
    assert 1 <= model.metadata["best_iteration"] <= 20
    second = train_lgbm(x, y, v, vy, config, static, 1, "tiny", num_threads=1)
    np.testing.assert_array_equal(model.booster.predict(v), second.booster.predict(v))
    save_lgbm(model, tmp_path)
    restored = load_lgbm(tmp_path)
    np.testing.assert_allclose(model.booster.predict(v), restored.booster.predict(v), atol=1e-12)
    record = evaluate(restored, panel, val_o, 1)
    np.testing.assert_allclose(record.predicted.reshape(-1), restored.booster.predict(v))
    assert_causal(lambda p, t, h: restored.predict_at(p.values[t-167:t+1], t, h), panel, 210, 1)
    with pytest.raises(ValueError):
        restored.predict_at(values[:168], 167, 1)
    with pytest.raises(ValueError):
        restored.predict_at(values[43:211], 210, 6)
