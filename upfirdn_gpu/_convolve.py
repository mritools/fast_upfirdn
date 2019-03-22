
import numpy as np
import scipy.ndimage as ndi

from upfirdn_gpu._util import get_array_module
from upfirdn_gpu._upfirdn import upfirdn

__all__ = ['convolve1d', 'convolve_separable']


def convolve1d(x, h, axis=-1, mode='reflect', cval=0, xp=None):
    """implementation of convolve1d with GPU support.

    Like scipy.ndimage.convolve1d, but always returns a floating point result.
    """
    # TODO: fix circular import requiring delayed import of upfirdn
    if xp is None:
        xp = get_array_module(x)[0]
    if x.shape[axis] < h.size // 2:
        raise ValueError("x.shape[axis] < h.size//2 unsupported")
    if mode == 'reflect':
        upfirdn_kwargs = dict(mode='symmetric')
    elif mode == 'mirror':
        upfirdn_kwargs = dict(mode='reflect')
    elif mode == 'nearest':
        upfirdn_kwargs = dict(mode='constant')
    elif mode == 'constant':
        if cval == 0:
            upfirdn_kwargs = dict(mode='zero')
        else:
            raise NotImplementedError("mode 'constant' not implemented")
    elif mode == 'wrap':
        upfirdn_kwargs = dict(mode='periodic')
    else:
        raise ValueError("unsupported mode: {}".format(mode))
    tmp = upfirdn(h, x, axis=axis, up=1, down=1, **upfirdn_kwargs)
    out_slices = [slice(None), ] * x.ndim
    nedge = h.size // 2
    out_slices[axis] = slice(nedge, nedge + x.shape[axis])
    return tmp[out_slices]


def convolve_separable(x, w, xp=None, **kwargs):
    """n-dimensional convolution via separable application of convolve1d

    currently a single 1d filter, w, is applied on all axes.
    w can also be a list of filters (equal in length to the number of axes)
    """
    axes = kwargs.pop('axes', range(x.ndim))
    if xp is None:
        xp = get_array_module(x)[0]
    if isinstance(w, xp.ndarray):
        w = [w, ] * len(axes)
    elif len(w) != len(axes):
        raise ValueError("user should supply one filter per axis")

    for ax, w0 in zip(axes, w):
        if xp == np:
            if xp.iscomplexobj(x):
                if not np.iscomplexobj(w0):
                    w0 = w0.astype(np.result_type(np.float32, x.dtype))
                tmp = ndi.convolve1d(x.real, w0.real, axis=ax, **kwargs)
                x = tmp + 1j * ndi.convolve1d(x.imag, w0.real, axis=ax,
                                              **kwargs)
            else:
                x = ndi.convolve1d(x, w0, axis=ax, **kwargs)
        else:
            x = convolve1d(x, w0, axis=ax, xp=xp, **kwargs)
    return x
