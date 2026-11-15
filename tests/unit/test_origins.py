import numpy as np
import pytest

from src.data.splits import build_split_config
from src.evaluation.origins import (
    assert_no_target_overlap, evaluation_origins, split_origins, train_origins, train_origins_joint,
)


@pytest.mark.parametrize("h,n_train,n_eval,last_train,last_eval", [
    (1, 504, 168, 670, 838), (6, 499, 163, 665, 833), (24, 481, 145, 647, 815)])
def test_primary_plan_boundaries(h, n_train, n_eval, last_train, last_eval):
    train = train_origins(671, h)
    val = evaluation_origins(671, 839, h)
    holdout = evaluation_origins(839, 1007, h)
    assert (len(train), train[0], train[-1]) == (n_train, 167, last_train)
    assert (len(val), val[0], val[-1]) == (n_eval, 671, last_eval)
    assert (len(holdout), holdout[0], holdout[-1]) == (n_eval, 839, last_eval + 168)


def test_joint_stride_empty_and_overlap():
    np.testing.assert_array_equal(train_origins_joint(671), np.arange(167, 648))
    np.testing.assert_array_equal(train_origins(671, 1, stride=3), np.arange(167, 671, 3))
    assert train_origins(100, 24).size == 0
    with pytest.raises(ValueError, match="overlap"):
        assert_no_target_overlap({"train": {1: np.array([5])}, "val": {6: np.array([0])}})
    for kwargs in ({"horizon": 0}, {"horizon": 1.5}, {"horizon": 1, "stride": 0}):
        with pytest.raises(ValueError):
            train_origins(671, **kwargs)


@pytest.mark.parametrize("primary,n", [(True, 1008), (False, 840)])
def test_split_target_disjointness_and_serialized_origins(primary, n):
    split = build_split_config(n, primary)
    origins = split_origins(split)
    assert_no_target_overlap(origins)
    for segment, horizon_origins in origins.items():
        bounds = split["fit" if segment == "train" else segment]
        for h, values in horizon_origins.items():
            assert np.all(values + h >= bounds[0])
            assert np.all(values + h <= bounds[1])
            assert split["origins"][segment][str(h)] == values.tolist()
