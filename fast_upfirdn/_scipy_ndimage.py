import numpy as np
import scipy.ndimage as ndi
from fast_upfirdn.cpu._upfirdn import upfirdn as upfirdn_cpu

from fast_upfirdn._util import get_array_module, array_on_device, have_cupy
if have_cupy:
    import cupy
    from fast_upfirdn.cupy._upfirdn import (
        _convolve1d as _convolve1d_gpu,
        _nearest_supported_float_dtype,
    )


__all__ = [
    # from scipy.ndimage API
    "convolve1d",
    "correlate1d",
    "gaussian_filter1d",
    "gaussian_filter",
    "uniform_filter1d",
    "uniform_filter",

    # not from SciPy API
    "convolve_separable",
]


def _check_axis(axis, rank):
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("invalid axis")
    return axis


def _get_output(output, arr, shape=None, xp=np):
    xp, on_gpu = get_array_module(arr)
    if on_gpu and isinstance(output, xp.ndarray):
        raise NotImplementedError("in-place operation not currently supported")
    if shape is None:
        shape = arr.shape
    if output is None:
        output = xp.zeros(shape, dtype=arr.dtype.name)
    elif type(output) in [type(type), type(cupy.zeros((4,)).dtype)]:
        output = xp.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = xp.typeDict[output]
        output = xp.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


# _gaussian_kernel1d is copied from SciPy
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def _fixup_dtypes(data, h):
    """Converts data and h to the nearest supported floating point type.

    Parameters
    ----------
    data, h : ndarray
        Input arrays.

    Returns
    -------
    data : ndarray
        ``data`` converted to the nearest supported dtype.
    data : ndarray
        ``h`` converted to the nearest supported dtype.
    dtype_out : np.dtype
        The dtype of the output array, ``np.result_type(data.dtype, h.dtype)``.
    """
    dtype_data, _ = _nearest_supported_float_dtype(data.dtype)
    if data.dtype != dtype_data:
        data = data.astype(dtype_data)

    # convert h to the same precision as data if there is a mismatch
    if data.real.dtype != h.real.dtype:
        if np.iscomplexobj(h):
            h_dtype = np.result_type(data.real.dtype, np.complex64)
        else:
            h_dtype = np.result_type(data.real.dtype, np.float32)
        h = h.astype(h_dtype)

    dtype_filter, _ = _nearest_supported_float_dtype(h.dtype)
    if h.dtype != dtype_filter:
        h = h.astype(dtype_filter)

    if np.iscomplexobj(h):
        # output is complex if filter is complex
        dtype_out = dtype_filter
    else:
        # for real filter, output dtype matches the data dtype
        dtype_out = dtype_data
    return data, h, dtype_out


def _normalize_sequence(arr, rank):
    """If arr is a scalar, create a sequence of length equal to the
    rank by duplicating the arr. If arr is a sequence,
    check if its length is equal to the length of array.
    """
    if hasattr(arr, "__iter__") and not isinstance(arr, str):
        normalized = list(arr)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to arr rank"
            raise RuntimeError(err)
    else:
        normalized = [arr] * rank
    return normalized


def _get_ndimage_mode_kwargs(mode, cval=0):
    if mode == "reflect":
        mode_kwargs = dict(mode="symmetric")
    elif mode == "mirror":
        mode_kwargs = dict(mode="reflect")
    elif mode == "nearest":
        mode_kwargs = dict(mode="edge")
    elif mode == "constant":
        mode_kwargs = dict(mode="constant", cval=cval)
    elif mode == "wrap":
        mode_kwargs = dict(mode="periodic")
    else:
        raise ValueError("unsupported mode: {}".format(mode))
    return mode_kwargs


def convolve1d(
    arr,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    xp=None,
    crop=True,  # TODO: remove crop argument (not in the ndimage API)
    # crop=False operates like np.convolve with mode='full'
):
    """Calculate a one-dimensional convolution along the given axis.

    see ``scipy.ndimage.convolve1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    xp, _ = get_array_module(arr)
    if xp == np:
        if crop:
            # ndi.convolve1d is faster than CPU-based upfirdn implementation
            return ndi.convolve1d(
                arr,
                weights,
                axis=axis,
                output=output,
                mode=mode,
                cval=cval,
                origin=origin,
            )
        else:
            mode_kwargs = _get_ndimage_mode_kwargs(mode, cval)
            if crop:
                offset = len(weights) // 2 + origin
            else:
                if origin != 0:
                    raise ValueError("uncropped case requires origin == 0")
                offset = 0
            out = upfirdn_cpu(
                weights,
                arr,
                offset=offset,
                axis=axis,
                crop=False,  # TODO: move crop from this function into upfirdn?
                up=1,
                down=1,
                **mode_kwargs,
            )
            sl_out = [slice(None)] * arr.ndim
            if crop:
                sl_out[axis] = slice(arr.shape[axis])
            return out[tuple(sl_out)]
    axis = _check_axis(axis, arr.ndim)

    # arr, weights, dtype_out = _fixup_dtypes(arr, weights)
    # if output is None:
    #     output = cupy.zeros(arr.shape, dtype=dtype_out)
    if output is not None:
        raise NotImplementedError("in-place operation not implemented")

    w_len_half = len(weights) // 2
    if crop:
        offset = w_len_half + origin
    else:
        if origin != 0:
            raise ValueError("uncropped case requires origin == 0")
        offset = 0
    if offset < -w_len_half or origin > ((len(weights) - 1) // 2):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= (len(weights)-1) // 2"
        )

    mode_kwargs = _get_ndimage_mode_kwargs(mode, cval)

    tmp = _convolve1d_gpu(
        weights,
        arr,
        axis=axis,
        offset=offset,
        crop=crop,
        out=output,
        **mode_kwargs
    )
    return tmp
    # out_slices = [slice(None), ] * arr.ndim
    # nedge = w_len_half
    # out_slices[axis] = slice(nedge, nedge + arr.shape[axis])
    # return tmp[tuple(out_slices)]


def correlate1d(
    arr,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    xp=None,
):
    """Calculate a one-dimensional correlation along the given axis.


    See ``scipy.ndimage.correlate1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return convolve1d(
        arr,
        weights,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        xp=xp,
    )


def uniform_filter1d(
    arr, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Calculate a one-dimensional uniform filter along the given axis.

    See ``scipy.ndimage.uniform_filter1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    xp, _ = get_array_module(arr)
    arr = xp.asarray(arr)
    # axis = _check_axis(axis, arr.ndim)
    if size < 1:
        raise RuntimeError("incorrect filter size")
    # output = _get_output(output, arr)  # TODO: add output support
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError("invalid origin")
    weights = xp.full(
        (size,), 1 / size, dtype=np.result_type(arr.real.dtype, np.float32)
    )
    return convolve1d(arr, weights, axis, output, mode, cval, origin, xp=xp)


def uniform_filter(
    arr, size=3, output=None, mode="reflect", cval=0.0, origin=0
):
    """Multi-dimensional uniform filter.

    See ``scipy.ndimage.uniform_filter``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    # xp, _ = get_array_module(arr)
    # arr = xp.asarray(arr)
    # # output = _ni_support._get_output(output, arr)  # TODO
    # dtype_arr = _nearest_supported_float_dtype(arr.dtype)
    # if arr.dtype != dtype_arr:
    #     arr = arr.astype(dtype_arr)
    # if output is None:
    #     output = cupy.zeros(arr.shape, dtype=dtype_arr)
    xp, _ = get_array_module(arr)
    output = _get_output(output, arr, xp=xp)
    sizes = _normalize_sequence(size, arr.ndim)
    origins = _normalize_sequence(origin, arr.ndim)
    modes = _normalize_sequence(mode, arr.ndim)
    axes = list(range(arr.ndim))
    axes = [
        (axes[ii], sizes[ii], origins[ii], modes[ii])
        for ii in range(len(axes))
        if sizes[ii] > 1
    ]
    if len(axes) > 0:

        # TODO
        # for axis, size, origin, mode in axes:
        #     uniform_filter1d(arr, int(size), axis, output, mode,
        #                      cval, origin)
        #     arr = output

        for axis, size, origin, mode in axes:
            arr = uniform_filter1d(
                arr, int(size), axis, output, mode, cval, origin
            )
        output = arr
    else:
        output[...] = arr[...]
    return output


def convolve_separable(x, w, xp=None, **kwargs):
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
            #if "origin" not in kwargs:
            #    kwargs["origin"] = -(len(w) // 2)
            x = convolve1d(x, w0, axis=ax, xp=xp, **kwargs)
    return x


def gaussian_filter1d(arr, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, _ = get_array_module(arr)
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = xp.asarray(weights)
    return correlate1d(arr, weights, axis, output, mode, cval, 0, xp=xp)


def gaussian_filter(arr, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, on_gpu = get_array_module(arr)
    arr = xp.asarray(arr)
    output = _get_output(output, arr, xp=xp)
    orders = _normalize_sequence(order, arr.ndim)
    sigmas = _normalize_sequence(sigma, arr.ndim)
    modes = _normalize_sequence(mode, arr.ndim)
    axes = list(range(arr.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(arr, sigma, axis, order, output,
                              mode, cval, truncate)
            arr = output
    else:
        output[...] = arr[...]
    return output
