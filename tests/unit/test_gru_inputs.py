import numpy as np

from src.data.panel import Panel
from src.features.beam_sampling import sample_beams
from src.features.gru_inputs import GRUBatches, GRUPreprocessor
from src.features.static_stats import compute_static
from tests.utils.perturb import perturb_after


def test_fit_boundary_masks_targets_sampling_and_future_canary(tmp_path):
    values = np.random.default_rng(17).normal(size=(220, 6, 4))
    values[170, 0, 0] = np.nan
    panel = Panel(values, np.arange(220), tuple(f"0_0_{i}" for i in range(6)), {})
    prep = GRUPreprocessor.fit(panel, 167)
    changed = GRUPreprocessor.fit(perturb_after(panel, 167), 167)
    for key in ("median", "mean", "scale"):
        np.testing.assert_array_equal(getattr(prep, key), getattr(changed, key))
    stats = compute_static(panel, 167)
    chosen = sample_beams(stats, 4)
    np.testing.assert_array_equal(chosen, sample_beams(stats, 4))
    assert len(set(chosen)) == 4
    batch = GRUBatches(panel, prep, "B", np.array([180]), np.array([0]))
    x, static, target = next(batch.batches(4))
    assert x.shape == (1, 168, 8) and static.shape == (1, 4) and target.shape == (1, 3)
    assert x[0, 157, 4] == 1  # origin180 history starts13; missing time170 -> offset157
    np.testing.assert_allclose(target.numpy()[0], (values[[181, 186, 204], 0, 0]-prep.mean[0])/prep.scale[0], rtol=1e-6)
    modified = GRUBatches(perturb_after(panel, 180), prep, "B", np.array([180]), np.array([0]))
    xx, ss, _ = next(modified.batches(4))
    np.testing.assert_array_equal(x.numpy(), xx.numpy())
    np.testing.assert_array_equal(static.numpy(), ss.numpy())
    assert GRUPreprocessor.fit(panel, 167, "A").transform(values, "A").shape == (220, 6, 2)
    prep.save(tmp_path / "prep.npz")
    np.testing.assert_array_equal(GRUPreprocessor.load(tmp_path / "prep.npz").mean, prep.mean)
