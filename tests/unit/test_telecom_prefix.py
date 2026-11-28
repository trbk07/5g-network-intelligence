"""Array-only prefix selection checks; no training."""

import numpy as np
import pytest

from scripts.prepare_telecom_prefix import prefix_stop


def test_prefix_excludes_cutoff_and_keeps_gaps():
    keys = np.array([0, 1, 4, 5, 8], dtype=np.uint32)
    assert prefix_stop(keys, 5, 10) == 3
    assert prefix_stop(keys, 0, 10) == 0
    assert prefix_stop(keys, 10, 10) == 5


@pytest.mark.parametrize("keys", [[0, 1, 1], [0, 3, 2], [-1, 0], [0, 10], []])
def test_invalid_entity_axes_fail_before_metric_read(keys):
    with pytest.raises(ValueError):
        prefix_stop(np.array(keys), 5, 10)
