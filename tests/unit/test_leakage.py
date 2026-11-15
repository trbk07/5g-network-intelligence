import numpy as np
import pytest

from src.data.panel import Panel
from src.evaluation.evaluator import history_at
from src.evaluation.metrics import assign_load_tiers
from src.models.baselines import Persistence, SeasonalNaive
from tests.utils.perturb import assert_causal, perturb_after


@pytest.fixture
def panel():
    return Panel(np.random.default_rng(17).normal(size=(220, 3, 4)),
                 np.arange(220), ("0_0_0", "0_0_1", "0_0_2"), {})


@pytest.mark.parametrize("model", [Persistence(), SeasonalNaive(24), SeasonalNaive(168)])
@pytest.mark.parametrize("h", [1, 6, 24])
def test_baseline_predictions_and_history_canary(panel, model, h):
    assert_causal(lambda p, t, horizon: model.predict(history_at(p, t), horizon), panel, 180, h)
    assert_causal(lambda p, t, horizon: history_at(p, t), panel, 180, h)


def test_canary_detects_deliberate_future_reader(panel):
    with pytest.raises(AssertionError):
        assert_causal(lambda p, t, h: p.values[t + h, :, 0], panel, 180, 6)


def test_load_tiers_do_not_read_outside_fit(panel):
    changed = perturb_after(panel, 167)
    np.testing.assert_array_equal(assign_load_tiers(panel.values[:168, :, 0]),
                                  assign_load_tiers(changed.values[:168, :, 0]))
