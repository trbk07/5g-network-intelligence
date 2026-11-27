"""Pure input-boundary tests; no external model load or fitting."""

import numpy as np
import pytest

from src.features.chronos2_inputs import history_tasks, training_prefix, native_tasks


def test_training_never_reads_post_fit_actuals():
    values = np.arange(840*3*4, dtype=float).reshape(840, 3, 4)
    expected = training_prefix(values, 335)
    values[336:] = np.inf
    actual = training_prefix(values, 335)
    assert actual.shape == (3, 4, 336)
    np.testing.assert_array_equal(actual, expected)
    values[:336] = -1
    assert not (actual == -1).all()  # Isolated copy, not a mutable panel view.


def test_history_preserves_four_channels_zero_and_native_missing():
    values = np.zeros((840, 3, 4))
    values[383, 2] = [1, 2, 3, np.nan]
    expected = history_tasks(values, 383, np.array([2, 0]))
    assert expected.shape == (2, 4, 168)
    np.testing.assert_array_equal(expected[0, :, -1], [1, 2, 3, np.nan])
    assert np.all(expected[1] == 0)
    values[384:] = np.inf
    np.testing.assert_array_equal(history_tasks(values, 383, np.array([2, 0])), expected)


def test_holdout_and_invalid_origins_rejected():
    values = np.zeros((840, 3, 4))
    with pytest.raises(ValueError):
        training_prefix(values, 671)
    with pytest.raises(ValueError):
        history_tasks(values, 672, np.array([0]))
    with pytest.raises(ValueError):
        history_tasks(values, 383, np.array([-1]))


def test_native_schema_does_not_supervise_future_covariate_values():
    array = np.arange(2*4*168, dtype=float).reshape(2, 4, 168)
    tasks = native_tasks(array)
    assert len(tasks) == 2
    assert set(tasks[0]) == {"target", "past_covariates"}
    assert set(tasks[0]["past_covariates"]) == {"DLThpVol", "DLThpTime", "MR_number"}
    np.testing.assert_array_equal(tasks[1]["target"], array[1, 0])
    np.testing.assert_array_equal(tasks[1]["past_covariates"]["MR_number"], array[1, 3])


def test_finetuning_entrypoint_refuses_local_before_importing_external_model(monkeypatch):
    from scripts import train_chronos2_lora

    def reject():
        raise RuntimeError("local forbidden")

    monkeypatch.setattr(train_chronos2_lora, "require_cloud_training", reject)
    with pytest.raises(RuntimeError, match="local forbidden"):
        train_chronos2_lora.main()
