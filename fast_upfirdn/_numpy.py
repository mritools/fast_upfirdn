"""Implementations of functions from the NumPy API via upfirdn.

"""
from functools import partial

import numpy as np

from fast_upfirdn._scipy_signal import upfirdn
from fast_upfirdn._util import get_array_module, have_cupy, check_device

if have_cupy:
    import cupy
    from fast_upfirdn.cupy._upfirdn import (
        _convolve1d as _convolve1d_gpu,
        _nearest_supported_float_dtype,
    )


__all__ = ["convolve", "correlate"]


def convolve(a, v, mode="full", xp=None):
    """see numpy.convolve

    The main difference in functionality is that this version only operates
    using np.float32, np.float64, np.complex64 and np.complex128.
    """
    xp, on_gpu = get_array_module(a, xp=xp)
    a, v = xp.array(a, copy=False, ndmin=1), xp.array(v, copy=False, ndmin=1)

    # make sure both arrays are on the CPU when xp=numpy or GPU when xp=cupy
    a, v = map(partial(check_device, xp=xp), [a, v])

    if len(a) < len(v):
        v, a = a, v
    if len(a) == 0:
        raise ValueError("a cannot be empty")
    if len(v) == 0:
        raise ValueError("v cannot be empty")
    if mode == "full":
        offset = 0
        size = len(a) + len(v) - 1
        crop = False
    elif mode == "same":
        offset = (len(v) - 1) // 2  # needed - 1 here to match NumPy
        size = len(a)
        crop = True
    elif mode == "valid":
        offset = len(v) - 1
        size = len(a) - len(v) + 1
        crop = True
    else:
        raise ValueError("unrecognized mode: {}".format(mode))
    if xp == np:
        out = upfirdn(v, a, offset=offset, mode="constant", cval=0, crop=crop)
    else:
        # TODO: only need this special case because crop=True doesn't work the
        #       same for upfirdn.
        out = _convolve1d_gpu(
            v, a, offset=offset, mode="constant", cval=0, crop=crop
        )
    return out[:size]


def correlate(a, v, mode="valid", xp=None):
    """see numpy.correlate

    The main difference in functionality is that this version only operates
    using np.float32, np.float64, np.complex64 and np.complex128.
    """
    xp, on_gpu = get_array_module(a, xp=xp)
    v = v[::-1]
    if xp.iscomplexobj(v):
        v = xp.conj(v)
    return convolve(a, v, mode=mode, xp=xp)
