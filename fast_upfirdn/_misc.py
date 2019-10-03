"""Misc convenience functions.
"""


import numpy as np
import scipy.ndimage as ndi

from fast_upfirdn._util import get_array_module
from fast_upfirdn._scipy_ndimage import convolve1d


__all__ = ["convolve_separable"]


def convolve_separable(x, w, *, xp=None, **kwargs):
    """n-dimensional convolution via separable application of convolve1d

    currently a single 1d filter, w, is applied on all axes.
    w can also be a list of filters (equal in length to the number of axes)
    """
    axes = kwargs.pop("axes", range(x.ndim))
    xp, _ = get_array_module(x)
    if isinstance(w, xp.ndarray):
        w = [w] * len(axes)
    elif len(w) != len(axes):
        raise ValueError("user should supply one filter per axis")

    for ax, w0 in zip(axes, w):
        if xp == np:
            if xp.iscomplexobj(x):
                if not np.iscomplexobj(w0):
                    w0 = w0.astype(np.result_type(np.float32, x.dtype))
                tmp = ndi.convolve1d(x.real, w0.real, axis=ax, **kwargs)
                x = tmp + 1j * ndi.convolve1d(
                    x.imag, w0.real, axis=ax, **kwargs
                )
            else:
                x = ndi.convolve1d(x, w0, axis=ax, **kwargs)
        else:
            x = convolve1d(x, w0, axis=ax, xp=xp, **kwargs)
    return x
