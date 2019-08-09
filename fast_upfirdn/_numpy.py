"""Implementations of functions from the NumPy API via upfirdn.

"""
import numpy as np

import cupy

from fast_upfirdn._util import get_array_module, have_cupy
if have_cupy:
    from fast_upfirdn.cupy._upfirdn import (
        _convolve1d as _convolve1d_gpu,
        _nearest_supported_float_dtype,
    )


__all__ = [
    "convolve",
    "correlate",
]


def convolve(a, v, mode='full'):
    """see numpy.convolve

    The main difference in functionality is that this version only operates
    using np.float32, np.float64, np.complex64 and np.complex128.
    """
    from fast_upfirdn import upfirdn
    xp, on_gpu = get_array_module(a, xp=None)
    a, v = xp.array(a, copy=False, ndmin=1), xp.array(v, copy=False, ndmin=1)
    if len(a) < len(v):
        v, a = a, v
    if len(a) == 0:
        raise ValueError('a cannot be empty')
    if len(v) == 0:
        raise ValueError('v cannot be empty')
    if mode == 'full':
        offset = 0
        size = len(a) + len(v) - 1
        crop = False
    elif mode == 'same':
        offset = (len(v) - 1) // 2  # needed - 1 here to match NumPy
        size = len(a)
        crop = True
    elif mode == 'valid':
        offset = len(v) - 1
        size = len(a) - len(v) + 1
        crop = True
    else:
        raise ValueError("unrecognized mode: {}".format(mode))
    if xp == np:
        out = upfirdn(
            v,
            a,
            offset=offset,
            mode='constant',
            cval=0,
            crop=crop)
    else:
        # TODO: only need this special case because crop=True doesn't work the
        #       same for upfirdn.
        out = _convolve1d_gpu(
            v,
            a,
            offset=offset,
            mode='constant',
            cval=0,
            crop=crop
        )
    return out[:size]


def correlate(a, v, mode="valid"):
    """see numpy.correlate

    The main difference in functionality is that this version only operates
    using np.float32, np.float64, np.complex64 and np.complex128.
    """
    v = v[::-1]
    if cupy.iscomplexobj(v):
        cupy.conj(v)
    return convolve(a, v, mode=mode)
