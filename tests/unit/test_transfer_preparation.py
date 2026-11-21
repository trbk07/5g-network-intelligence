"""Cross-platform metadata regression; no model fitting."""

import json

import numpy as np

from scripts.prepare_transfer_data import entity_identity


def test_exact_complete_hours_preserve_gaps_and_missing():
    import pandas as pd
    from scripts.prepare_transfer_data import complete_hour_means

    times = pd.date_range("2025-01-01", periods=16, freq="15min", tz="UTC")
    values = np.arange(16, dtype=float)
    values[5] = np.nan
    frame = pd.DataFrame({"Timestamp": times, "5G RB utilization": values})
    # Hour 1 has a missing target; hour 2 an invalid offset. Neither is manufactured.
    frame.loc[8, "Timestamp"] += pd.Timedelta(seconds=1)
    hourly = complete_hour_means(frame)
    assert list(hourly.index.hour) == [0, 3]
    np.testing.assert_allclose(hourly["mean"], [1.5, 13.5])


def test_numpy_group_key_is_json_native():
    entity = entity_identity((np.str_("Site 107"), np.int64(1)))
    assert type(entity[0]) is str and type(entity[1]) is int
    assert json.loads(json.dumps({"entity": entity})) == {"entity": ["Site 107", 1]}
