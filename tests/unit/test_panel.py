import numpy as np
import pandas as pd
import pytest

from src.data.panel import build_panel, load_panel, save_panel
from src.data.schema import KPIS, parse_beam_id


@pytest.fixture
def files(tmp_path):
    result = {}
    for kpi in KPIS:
        path = tmp_path / f"{kpi}.csv"
        pd.DataFrame({"Time": [1, 0], "10_0_0": [np.nan, 2],
                      "2_0_0": [3, 0]}).to_csv(path, index=False)
        result[kpi] = path
    return result


def test_sort_nan_zero_roundtrip(files, tmp_path):
    panel = build_panel(files)
    assert panel.beam_ids == ("2_0_0", "10_0_0")
    assert panel.time.tolist() == [0, 1]
    assert panel.values[0, 0, 0] == 0
    assert np.isnan(panel.values[1, 1, 0])
    save_panel(panel, tmp_path / "panel.npz")
    restored = load_panel(tmp_path / "panel.npz")
    np.testing.assert_array_equal(panel.values, restored.values)
    assert panel.metadata == restored.metadata
    assert restored.beam_ids == panel.beam_ids


def test_outer_alignment(files):
    pd.DataFrame({"Time": [0, 2], "2_0_0": [7, 9]}).to_csv(files["DLPRB"], index=False)
    panel = build_panel(files)
    assert panel.time.tolist() == [0, 1, 2]
    assert panel.values[2, 0, 0] == 9
    assert np.isnan(panel.values[1, 0, 0])
    assert np.isnan(panel.values[2, 0, 1])


@pytest.mark.parametrize("contents", [
    "Time,0_0_0\n0,1\n0,2\n", "Time,0_0_0,0_0_0\n0,1,2\n",
    "Time,0_0_0\n0.5,1\n", "Time,0_0_0\n0,inf\n",
    "Time,bad\n0,1\n", "Time,0_0_0\n0,invalid\n",
])
def test_bad_schema_rejected(files, contents):
    files["DLPRB"].write_text(contents)
    with pytest.raises(ValueError):
        build_panel(files)


def test_beam_parser():
    assert parse_beam_id("29_2_31") == (29, 2, 31)
    with pytest.raises(ValueError):
        parse_beam_id("bs1_cell2_beam3")


def test_actual_release_unnamed_time(files):
    files["DLPRB"].write_text(",10_0_0,2_0_0\n0,2,0\n1,,3\n")
    panel = build_panel(files)
    assert panel.values[0, 0, 0] == 0
    assert panel.metadata["source_audit"]["DLPRB"]["unnamed_time_column"]
