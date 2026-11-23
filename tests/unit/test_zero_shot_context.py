"""History slicing only; no pretrained model loading or fitting."""

import numpy as np
import pytest

from src.features.zero_shot_context import target_context


def test_window_preserves_zeros_nans_and_target_only():
    values=np.ones((200,2,4))
    values[0,1,0]=0
    values[1,1,0]=np.nan
    context=target_context(values,167,np.array([1]))
    assert context.shape==(1,168) and context.dtype==np.float32
    assert context[0,0]==0 and np.isnan(context[0,1])
    values[:,1,1:]=999
    np.testing.assert_array_equal(context,target_context(values,167,np.array([1])))


def test_future_perturbation_cannot_change_context():
    values=np.zeros((200,2,4))
    expected=target_context(values,175,np.array([0,1]))
    values[176:]=10000
    np.testing.assert_array_equal(expected,target_context(values,175,np.array([0,1])))


@pytest.mark.parametrize("origin",[166,200])
def test_invalid_origin_fails(origin):
    with pytest.raises(ValueError):
        target_context(np.zeros((200,1,4)),origin,np.array([0]))
