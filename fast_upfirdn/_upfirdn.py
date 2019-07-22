
import numpy as np
from fast_upfirdn.cpu import upfirdn as upfirdn_cpu
from fast_upfirdn.cpu._upfirdn_apply import _output_len as upfirdn_out_len
from fast_upfirdn._util import get_array_module, array_on_device, have_cupy

try:
    import cupy
    have_cupy = True
except ImportError:
    have_cupy = False

if have_cupy:
    from fast_upfirdn.cupy import upfirdn as upfirdn_cupy


__all__ = ['upfirdn', 'upfirdn_out_len']


def get_array_module(arr, xp=None):
    """ Check if the array is a cupy GPU array and return the array module.

    Parameters
    ----------
    arr : numpy.ndarray or cupy.core.core.ndarray
        The array to check.

    Returns
    -------
    array_module : python module
        This will be cupy when on_gpu is True and numpy otherwise.
    on_gpu : bool
        Boolean indicating whether the array is on the GPU.
    """
    if xp is None:
        if isinstance(arr, np.ndarray) or not have_cupy:
            return np, False
        else:
            xp = cupy.get_array_module(arr)
            return xp, (xp != np)
    else:
        return xp, (xp != np)


def array_on_device(arr, xp):
    if xp == np:
        if have_cupy and hasattr(arr, '__cuda_array_interface__'):
            # copy back from GPU
            return arr.get()
    return xp.asarray(arr)


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    xp=None,
    prepadded=False,
    out=None,
    h_size_orig=None,
    mode="zero",
    cval=0,
    origin=0,
    crop=False,
    take=None,
):
    # If xp was unspecified, determine it from x
    xp, on_gpu = get_array_module(x, xp)

    # make sure both arrays are on the CPU when xp=numpy or GPU when xp=cupy
    x, h = [array_on_device(arr, xp) for arr in [x, h]]

    upfirdn_kwargs = dict(up=up, down=down, axis=axis, mode=mode, cval=cval,
                          origin=origin, crop=int(crop), take=take)

    if on_gpu:
        y = upfirdn_cupy(
            h,
            x,
            prepadded=prepadded,
            h_size_orig=h_size_orig,
            out=out,
            **upfirdn_kwargs,
        )
    else:
        if prepadded:
            raise ValueError("prepadded not supported on the CPU")
        if out is not None:
            raise ValueError("preallocated out array not supported on the CPU")
        upfirdn_kwargs.pop('take')
        y = upfirdn_cpu(
            h,
            x,
            **upfirdn_kwargs,
        )
    return y
