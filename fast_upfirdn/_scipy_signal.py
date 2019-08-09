"""Implementation of functions from the scipy.signal API.

Currently this is just ``upfirdn``.

Also defines ``upfirdn_out_len`` which is not part of the public scipy API.

"""
from fast_upfirdn.cpu import upfirdn as upfirdn_cpu
from fast_upfirdn.cpu._upfirdn_apply import _output_len as upfirdn_out_len
from fast_upfirdn._util import get_array_module, array_on_device, have_cupy

if have_cupy:
    from fast_upfirdn.cupy import upfirdn as upfirdn_cupy


__all__ = ['upfirdn', 'upfirdn_out_len']


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
    offset=0,
    crop=False,
    take=None,
):
    """

    See ``scipy.signal.upfirdn``. This version supports some additional
    keyword arguments.

    Additional Parameters
    ---------------------
    xp : module or None
        The array module (``cupy`` or ``numpy``). If not provided, it is
        inferred from the type of ``x``.
    prepadded : bool, optional
        If this is True, it is assumed that the internal computation
        ``h = _pad_h(h, up=up)`` has already been performed on ``h``.
    out : ndarray
        TODO
    h_size_orig : int, optional
        TODO
    mode : str, optional
        The signal extension mode used at the boundaries.
    cval : float, optional
        The constant value used when ``mode == "constant"``.
    offset : int, optional
        TODO
    crop : bool, optional
        TODO
    take : int or None, optional
        TODO


    """

    # If xp was unspecified, determine it from x
    xp, on_gpu = get_array_module(x, xp)

    # make sure both arrays are on the CPU when xp=numpy or GPU when xp=cupy
    x, h = [array_on_device(arr, xp) for arr in [x, h]]

    upfirdn_kwargs = dict(up=up, down=down, axis=axis, mode=mode, cval=cval,
                          offset=offset, crop=int(crop), take=take)

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
