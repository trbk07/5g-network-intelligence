"""Non-training checks for causal external segment construction."""

import numpy as np
import pytest

from scripts.prepare_cesnet_pretraining import split_segments


def test_gaps_are_not_bridged():
    hours = np.r_[np.arange(200), np.arange(201, 401)]
    result = split_segments(hours, np.ones(400))
    assert [len(h) for h, _ in result] == [200, 200]
    assert result[0][0][-1] == 199
    assert result[1][0][0] == 201


def test_invalid_values_break_windows():
    signal = np.ones(400)
    signal[200] = np.nan
    assert [len(h) for h, _ in split_segments(np.arange(400), signal)] == [200, 199]


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        split_segments(np.r_[np.arange(200), 199], np.ones(201))
