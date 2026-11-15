"""Future perturbations used only for testing causality."""
import numpy as np

from src.data.panel import Panel


def perturb_after(panel: Panel, origin: int, seed: int = 17) -> Panel:
    """Replace all observations after origin, including missing values, with noise."""
    values = panel.values.copy()
    rng = np.random.default_rng(seed)
    values[origin + 1:] = rng.normal(1000, 100, values[origin + 1:].shape)
    return Panel(values, panel.time.copy(), panel.beam_ids, dict(panel.metadata))


def assert_causal(predict, panel: Panel, origin: int, horizon: int) -> None:
    """Assert a panel-to-output adapter is invariant to all future observations."""
    before = predict(panel, origin, horizon)
    after = predict(perturb_after(panel, origin), origin, horizon)
    np.testing.assert_array_equal(before, after)
