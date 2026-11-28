"""Array/forward-only tests; no fit, backward or optimizer calls."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.data.panel import Panel
from src.features.cell_inputs import CellBatches
from src.features.gru_inputs import GRUPreprocessor
from src.features.partial_cell_inputs import PartialCellBatches


def fixture():
    values = np.broadcast_to(np.arange(504.)[:, None, None], (504, 32, 4)).copy()
    panel = Panel(values, np.arange(504), tuple(f"0_0_{b}" for b in range(32)), {})
    return panel, GRUPreprocessor.fit(panel, 335, "B")


def test_partial_labels_are_masked_and_future_backing_values_excluded():
    panel, pre = fixture()
    batcher = PartialCellBatches(panel, pre, [311, 312, 329, 334], 335)
    a = next(batcher.batches(4))
    expected = torch.tensor([[312., 317., 335.], [313., 318., float("nan")],
                             [330., 335., float("nan")], [335., float("nan"), float("nan")]])
    torch.testing.assert_close(a[2][:, 0], expected, equal_nan=True)
    assert batcher.values.shape[0] == 336 and batcher.target.shape[0] == 336
    poisoned = panel.values.copy()
    poisoned[336:] = 1e12
    b = next(PartialCellBatches(replace(panel, values=poisoned), pre, [311, 312, 329, 334], 335).batches(4))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, equal_nan=True, atol=0, rtol=0)


def test_complete_cases_exactly_match_historical_batcher():
    panel, pre = fixture()
    origins = np.arange(167, 312)
    historical = CellBatches(panel, pre, origins)
    partial = PartialCellBatches(panel, pre, origins, 335)
    for a, b in zip(historical.batches(32, 17), partial.batches(32, 17)):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_all_valid_short_horizon_labels_retained_once():
    panel, pre = fixture()
    batcher = PartialCellBatches(panel, pre, np.arange(167, 335), 335)
    counts = torch.zeros(3, dtype=torch.int64)
    for _, _, target in batcher.batches(17):
        counts += torch.isfinite(target).sum((0, 1))
    assert counts.tolist() == [168*32, 163*32, 145*32]
    # The same counts scale to all 2,880 original beams without reading their values.
    assert int(counts.sum())*90 == 1370880


@pytest.mark.parametrize("origins,end", [([334.5], 335), ([334, 334], 335), ([335], 335),
                                        ([166], 335), ([334], 504), ([334], True)])
def test_invalid_boundaries_fail(origins, end):
    panel, pre = fixture()
    with pytest.raises(ValueError):
        PartialCellBatches(panel, pre, origins, end)
