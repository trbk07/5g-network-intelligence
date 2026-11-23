"""Causal target-only windows, with zeros and missingness preserved."""

import numpy as np


def target_context(values, origin, beams):
    values=np.asarray(values)
    beams=np.asarray(beams)
    if (values.ndim!=3 or values.shape[2]!=4 or not isinstance(origin,(int,np.integer))
            or origin<167 or origin>=len(values) or beams.ndim!=1 or not beams.size
            or not np.issubdtype(beams.dtype,np.integer) or beams.min()<0 or beams.max()>=values.shape[1]):
        raise ValueError("Valid absolute origin, beam indices and 168 historical rows required")
    result=values[origin-167:origin+1,beams,0].T.astype(np.float32,copy=True)
    if np.isinf(result).any():
        raise ValueError("Infinite input is invalid; missing observations must use NaN")
    return result
