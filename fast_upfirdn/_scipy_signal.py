"""Implementation of functions from the scipy.signal API.

Currently this is just ``upfirdn`` and ``resample_poly``.

Also defines ``upfirdn_out_len`` which is not part of the public scipy API.

"""
from functools import partial
import sys

from fast_upfirdn.cpu import upfirdn as upfirdn_cpu
from fast_upfirdn.cpu._upfirdn import _upfirdn_modes
from fast_upfirdn.cpu._upfirdn_apply import _output_len as upfirdn_out_len
from fast_upfirdn._util import (
    get_array_module,
    have_cupy,
    check_device,
    profile
)

if sys.version_info >= (3, 5):
    from math import gcd
else:
    from fractions import gcd

if have_cupy:
    from fast_upfirdn.cupy import upfirdn as upfirdn_cupy

__all__ = ["upfirdn", "upfirdn_out_len", "resample_poly"]


@profile
def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    mode="zero",
    cval=0,
    *,
    xp=None,
    prepadded=False,
    out=None,
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
    x, h = map(partial(check_device, xp=xp), [x, h])

    upfirdn_kwargs = dict(
        up=up,
        down=down,
        axis=axis,
        mode=mode,
        cval=cval,
        offset=offset,
        crop=int(crop),
        take=take,
    )

    if on_gpu:
        y = upfirdn_cupy(h, x, prepadded=prepadded, out=out, **upfirdn_kwargs)
    else:
        if prepadded:
            raise ValueError("prepadded not supported on the CPU")
        if out is not None:
            raise ValueError("preallocated out array not supported on the CPU")
        y = upfirdn_cpu(h, x, **upfirdn_kwargs)
    return y


def _resample_poly_window(up, down, window=("kaiser", 5.0)):
    """Design a linear-phase low-pass FIR filter for resample_poly."""
    from scipy.signal import firwin

    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)
    half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
    h = firwin(2 * half_len + 1, f_c, window=window)
    return h, half_len


def resample_poly(
    x,
    up,
    down,
    axis=0,
    window=("kaiser", 5.0),
    padtype="constant",
    cval=None,
    xp=None,
):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `np.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the axis.

    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    xp : module or None
        The array module (``cupy`` or ``numpy``). If not provided, it is
        inferred from the type of ``x``.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `scipy.signal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `scipy.signal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `scipy.signal.get_window`
    and `scipy.signal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    By default, the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> from scipy import signal

    >>> x = np.linspace(0, 10, 20, endpoint=False)
    >>> y = np.cos(-x**2/6.0)
    >>> f_fft = signal.resample(y, 100)
    >>> f_poly = signal.resample_poly(y, 100, 20)
    >>> xnew = np.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
    >>> plt.plot(x, y, 'ko-')
    >>> plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()

    This default behaviour can be changed by using the padtype option:

    >>> import numpy as np
    >>> from scipy import signal

    >>> N = 5
    >>> x = np.linspace(0, 1, N, endpoint=False)
    >>> y = 2 + x**2 - 1.7*np.sin(x) + .2*np.cos(11*x)
    >>> y2 = 1 + x**3 + 0.1*np.sin(x) + .1*np.cos(11*x)
    >>> Y = np.stack([y, y2], axis=-1)
    >>> up = 4
    >>> xr = np.linspace(0, 1, N*up, endpoint=False)

    >>> y2 = signal.resample_poly(Y, up, 1, padtype='constant')
    >>> y3 = signal.resample_poly(Y, up, 1, padtype='mean')
    >>> y4 = signal.resample_poly(Y, up, 1, padtype='line')

    >>> import matplotlib.pyplot as plt
    >>> for i in [0,1]:
    ...     plt.figure()
    ...     plt.plot(xr, y4[:,i], 'g.', label='line')
    ...     plt.plot(xr, y3[:,i], 'y.', label='mean')
    ...     plt.plot(xr, y2[:,i], 'r.', label='constant')
    ...     plt.plot(x, Y[:,i], 'k-')
    ...     plt.legend()
    >>> plt.show()

    """
    # If xp was unspecified, determine it from x
    xp, on_gpu = get_array_module(x, xp)

    if not on_gpu:
        from scipy.signal import resample_poly

        return resample_poly(
            x, up, down, axis=axis, window=window, padtype=padtype, cval=cval
        )

    x = xp.asarray(x)
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    if cval is not None and padtype != "constant":
        raise ValueError("cval has no effect when padtype is ", padtype)

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_in = x.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list, xp.ndarray)):
        window = xp.array(window)  # use array to force a copy (we modify it)
        if window.ndim > 1:
            raise ValueError("window must be 1-D")
        half_len = (window.size - 1) // 2
        h = window
    else:
        # Design a linear-phase low-pass FIR filter
        h, half_len = _resample_poly_window(up, down, window=window)
        h = xp.asarray(h)
    h *= up

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while (
        upfirdn_out_len(len(h) + n_pre_pad + n_post_pad, n_in, up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1
    h = xp.concatenate(
        (
            xp.zeros(n_pre_pad, dtype=h.dtype),
            h,
            xp.zeros(n_post_pad, dtype=h.dtype),
        )
    )
    n_pre_remove_end = n_pre_remove + n_out

    # Remove background depending on the padtype option
    funcs = {
        "mean": xp.mean,
        # 'median': xp.median,   # TODO: needs cupy.median implementation
        "minimum": xp.amin,
        "maximum": xp.amax,
    }
    upfirdn_kwargs = {'mode': 'constant', 'cval': 0}
    if padtype in funcs:
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in _upfirdn_modes:
        upfirdn_kwargs = {'mode': padtype}
        if padtype == 'constant':
            if cval is None:
                cval = 0
            upfirdn_kwargs['cval'] = cval
    else:
        raise ValueError(
            'padtype must be one of: maximum, mean, median, minimum, ' +
            ', '.join(_upfirdn_modes))

    if padtype in funcs:
        x = x - background_values

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis=axis, xp=xp, **upfirdn_kwargs)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y_keep = y[tuple(keep)]

    # Add background back
    if padtype in funcs:
        y_keep += background_values

    return y_keep
