import json

import numpy as np
import pytest
import torch

from src.data.panel import Panel
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.models.gru import GRUNet
from src.models.gru_train import masked_l1, train_gru
from src.models.gru_forecaster import GRUForecaster
from tests.utils.perturb import assert_causal

pytestmark = pytest.mark.cloud_training


def test_resume_matches_uninterrupted_training_and_cpu_guard(tmp_path):
    torch.set_num_threads(1)
    panel = Panel(np.random.default_rng(17).uniform(size=(220, 2, 4)),
                  np.arange(220), ("0_0_0", "0_0_1"), {})
    prep = GRUPreprocessor.fit(panel, 167, "A")
    train = GRUBatches(panel, prep, "A", np.array([167, 170]), np.arange(2))
    val = GRUBatches(panel, prep, "A", np.array([180]), np.arange(2))
    config = {"learning_rate": .001, "max_epochs": 2, "batch_size": 2,
              "gradient_clip": 1., "early_stopping_patience": 5}
    torch.manual_seed(17)
    full = GRUNet(2, 8, 2, .1)
    train_gru(full, train, val, config, tmp_path/"full", 17, allow_cpu_test=True)
    torch.manual_seed(17)
    partial = GRUNet(2, 8, 2, .1)
    train_gru(partial, train, val, {**config, "max_epochs": 1}, tmp_path/"resume", 17,
              allow_cpu_test=True)
    restored = GRUNet(2, 8, 2, .1)
    train_gru(restored, train, val, config, tmp_path/"resume", 17, resume=True, allow_cpu_test=True)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="CPU"):
        train_gru(restored, train, val, config, tmp_path/"guard", 17)
    loss, count = masked_l1(torch.tensor([[1., 9.]]), torch.tensor([[0., float("nan")]]))
    assert loss == 1 and count == 1
    prep.save(tmp_path/"full/preprocessor.npz")
    (tmp_path/"full/meta.json").write_text(json.dumps({"variant": "A", "seed": 17,
        "architecture": {"hidden_size": 8, "num_layers": 2, "dropout": .1}}))
    predictor = GRUForecaster(tmp_path/"full")
    batch = predictor.predict_origins(panel, np.array([180]))
    single = predictor.predict_all(panel.values[13:181], 180)
    np.testing.assert_allclose(batch[0], single, rtol=1e-6, atol=1e-6)
    assert_causal(lambda p,t,h: predictor.predict_at(p.values[t-167:t+1],t,h),panel,180,6)
