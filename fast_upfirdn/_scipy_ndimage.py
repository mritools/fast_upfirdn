"""Functions based on separable convolution found in scipy.ndimage

These functions are intended to operate exactly like their SciPy counterparts
aside from the following differences:

1.) Computations are done in the nearest floating point precision (single or
    double) for the input dtype.
    (``scipy.ndimage`` does all convolutions in double precision)
2.) Complex-valued inputs (complex64 or complex128) are supported.
    (``scipy.ndimage`` does not support complex-valued inputs)
3.) convolve1d currently has a ``crop`` kwarg. If set to False, a full
    convolution instead of one truncated to the size of the input is given.
4.) All functions have an ``xp`` keyword-only argument that can be set to
    either the NumPy or CuPy module or None. If None, the array backend to use
    is determined based on the type of the input array.
5.) In-place operation via ``output`` is not currently supported for GPU
    arrays.

"""
import numpy as np
import scipy.ndimage as ndi
from fast_upfirdn.cpu._upfirdn import upfirdn as upfirdn_cpu

from fast_upfirdn._util import get_array_module, have_cupy, check_device

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
    "prewitt",
    "sobel",
    "generic_laplace",
    "laplace",
    "gaussian_laplace",
    "generic_gradient_magnitude",
    "gaussian_gradient_magnitude",
    # not from SciPy API
    "convolve_separable",
]


def _check_axis(axis, rank):
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("invalid axis")
    return axis


def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)


def _get_output(output, arr, shape=None, xp=np):
    xp, on_gpu = get_array_module(arr, xp)
    if on_gpu and isinstance(output, xp.ndarray):
        # TODO: support in-place output on GPU
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
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
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
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
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
    *,
    xp=None,
    crop=True,  # if False, will get a "full" convolution instead
    # crop=False operates like np.convolve with mode='full'
):
    """Calculate a one-dimensional convolution along the given axis.

    see ``scipy.ndimage.convolve1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    if _invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )

    xp, _ = get_array_module(arr)
    arr = check_device(arr, xp)

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

    if output is not None:
        raise NotImplementedError("in-place operation not implemented")

    w_len_half = len(weights) // 2
    if crop:
        offset = w_len_half + origin
    else:
        if origin != 0:
            raise ValueError("uncropped case requires origin == 0")
        offset = 0

    mode_kwargs = _get_ndimage_mode_kwargs(mode, cval)

    tmp = _convolve1d_gpu(
        weights,
        arr,
        axis=axis,
        offset=offset,
        crop=crop,
        out=output,
        **mode_kwargs,
    )
    return tmp


def correlate1d(
    arr,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    *,
    xp=None,
):
    """Calculate a one-dimensional correlation along the given axis.


    See ``scipy.ndimage.correlate1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    if _invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )

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
    arr,
    size,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    xp=None,
):
    """Calculate a one-dimensional uniform filter along the given axis.

    See ``scipy.ndimage.uniform_filter1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    xp, _ = get_array_module(arr, xp)
    arr = xp.asarray(arr)
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
    arr, size=3, output=None, mode="reflect", cval=0.0, origin=0, *, xp=None
):
    """Multi-dimensional uniform filter.

    See ``scipy.ndimage.uniform_filter``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    xp, _ = get_array_module(arr, xp)
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

        if False:
            # TODO: add output != None support
            for axis, size, origin, mode in axes:
                uniform_filter1d(
                    arr, int(size), axis, output, mode, cval, origin
                )
                arr = output
        else:
            for axis, size, origin, mode in axes:
                arr = uniform_filter1d(
                    arr, int(size), axis, None, mode, cval, origin
                )
            output[...] = arr[...]
    else:
        output[...] = arr[...]
    return output


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
            # if "origin" not in kwargs:
            #    kwargs["origin"] = -(len(w) // 2)
            x = convolve1d(x, w0, axis=ax, xp=xp, **kwargs)
    return x


def gaussian_filter1d(
    arr,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    xp=None,
):
    """One-dimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter1d``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, _ = get_array_module(arr, xp)
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = xp.asarray(weights)
    return correlate1d(arr, weights, axis, output, mode, cval, 0, xp=xp)


def gaussian_filter(
    arr,
    sigma,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    xp=None,
):
    """Multidimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, on_gpu = get_array_module(arr, xp)
    arr = xp.asarray(arr)
    output = _get_output(output, arr, xp=xp)
    orders = _normalize_sequence(order, arr.ndim)
    sigmas = _normalize_sequence(sigma, arr.ndim)
    modes = _normalize_sequence(mode, arr.ndim)
    axes = list(range(arr.ndim))
    axes = [
        (axes[ii], sigmas[ii], orders[ii], modes[ii])
        for ii in range(len(axes))
        if sigmas[ii] > 1e-15
    ]
    if len(axes) > 0:
        if False:
            # TODO: add support for output argument
            for axis, sigma, order, mode in axes:
                gaussian_filter1d(
                    arr,
                    sigma,
                    axis,
                    order,
                    output,
                    mode,
                    cval,
                    truncate,
                    xp=xp,
                )
                arr = output
        else:
            for axis, sigma, order, mode in axes:
                arr = gaussian_filter1d(
                    arr, sigma, axis, order, None, mode, cval, truncate, xp=xp
                )
            output[...] = arr[...]
    else:
        output[...] = arr[...]
    return output


def prewitt(arr, axis=-1, output=None, mode="reflect", cval=0.0, *, xp=None):
    """Apply a Prewitt filter.

    See ``scipy.ndimage.prewitt``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, on_gpu = get_array_module(arr, xp)
    arr = check_device(arr, xp)
    axis = _check_axis(axis, arr.ndim)
    output = _get_output(output, arr)
    modes = _normalize_sequence(mode, arr.ndim)
    filt1 = [-1, 0, 1]
    filt2 = [1, 1, 1]
    if on_gpu:
        filt1, filt2 = map(cupy.asarray, [filt1, filt2])
    if not on_gpu:
        # TODO: enable output support in correlate1d on the GPU

        correlate1d(arr, filt1, axis, output, modes[axis], cval, 0, xp=xp)
        axes = [ii for ii in range(arr.ndim) if ii != axis]
        for ii in axes:
            correlate1d(output, filt2, ii, output, modes[ii], cval, 0, xp=xp)
    else:
        output = correlate1d(
            arr, filt1, axis, None, modes[axis], cval, 0, xp=xp
        )
        axes = [ii for ii in range(arr.ndim) if ii != axis]
        for ii in axes:
            output = correlate1d(
                output, filt2, ii, None, modes[ii], cval, 0, xp=xp
            )
    return output


def sobel(arr, axis=-1, output=None, mode="reflect", cval=0.0, *, xp=None):
    """Apply a sobel filter.

    See ``scipy.ndimage.sobel``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, on_gpu = get_array_module(arr, xp)
    arr = check_device(arr, xp)
    output = _get_output(output, arr)
    modes = _normalize_sequence(mode, arr.ndim)
    filt1 = [-1, 0, 1]
    filt2 = [1, 2, 1]
    if on_gpu:
        filt1, filt2 = map(cupy.asarray, [filt1, filt2])

    if not on_gpu:
        # TODO: enable output support in correlate1d on the GPU
        correlate1d(arr, filt1, axis, output, modes[axis], cval, 0, xp=xp)
        axes = [ii for ii in range(arr.ndim) if ii != axis]
        for ii in axes:
            correlate1d(output, filt2, ii, output, modes[ii], cval, 0, xp=xp)
    else:
        output = correlate1d(
            arr, filt1, axis, None, modes[axis], cval, 0, xp=xp
        )
        axes = [ii for ii in range(arr.ndim) if ii != axis]
        for ii in axes:
            output = correlate1d(
                output, filt2, ii, None, modes[ii], cval, 0, xp=xp
            )

    return output


def generic_laplace(
    arr,
    derivative2,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
    *,
    xp=None,
):
    """
    N-dimensional Laplace filter using a provided second derivative function.

    See ``scipy.ndimage.generic_laplace``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    if extra_keywords is None:
        extra_keywords = {}
    xp, _ = get_array_module(arr, xp)
    arr = check_device(arr, xp)
    output = _get_output(output, arr)
    axes = list(range(arr.ndim))
    if len(axes) > 0:
        modes = _normalize_sequence(mode, len(axes))
        if False:
            # TODO: enable branch once in-place output is supported
            derivative2(
                arr,
                axes[0],
                output,
                modes[0],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
        else:
            output = derivative2(
                arr,
                axes[0],
                None,
                modes[0],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
        for ii in range(1, len(axes)):
            if False:
                # TODO: enable branch once in-place output is supported
                tmp = derivative2(
                    arr,
                    axes[ii],
                    output.dtype,
                    modes[ii],
                    cval,
                    *extra_arguments,
                    **extra_keywords,
                )
            else:
                tmp = derivative2(
                    arr,
                    axes[ii],
                    None,
                    modes[ii],
                    cval,
                    *extra_arguments,
                    **extra_keywords,
                )
            output += tmp
    else:
        output[...] = arr[...]
    return output


def laplace(arr, output=None, mode="reflect", cval=0.0, *, xp=None):
    """N-dimensional Laplace filter based on approximate second derivatives.

    See ``scipy.ndimage.laplace``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, _ = get_array_module(arr, xp)
    arr = check_device(arr, xp)

    def derivative2(arr, axis, output, mode, cval, xp=xp):
        h = xp.asarray([1, -2, 1])
        return correlate1d(arr, h, axis, output, mode, cval, 0, xp=xp)

    return generic_laplace(arr, derivative2, output, mode, cval, xp=xp)


def gaussian_laplace(
    arr, sigma, output=None, mode="reflect", cval=0.0, *, xp=None, **kwargs
):
    """Multidimensional Laplace filter using gaussian second derivatives.

    See ``scipy.ndimage.gaussian_laplace``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    xp, _ = get_array_module(arr, xp)
    arr = check_device(arr, xp)

    def derivative2(arr, axis, output, mode, cval, sigma, xp=xp, **kwargs):
        order = [0] * arr.ndim
        order[axis] = 2
        return gaussian_filter(
            arr, sigma, order, output, mode, cval, xp=xp, **kwargs
        )

    return generic_laplace(
        arr,
        derivative2,
        output,
        mode,
        cval,
        extra_arguments=(sigma,),
        extra_keywords=kwargs,
        xp=xp,
    )


def generic_gradient_magnitude(
    arr,
    derivative,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
    *,
    xp=None,
):
    """Gradient magnitude using a provided gradient function.

    See ``scipy.ndimage.generic_gradient_magnitude``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.

    """
    if extra_keywords is None:
        extra_keywords = {}
    xp, _ = get_array_module(arr, xp)
    arr = check_device(arr, xp)
    output = _get_output(output, arr)
    axes = list(range(arr.ndim))
    if len(axes) > 0:
        modes = _normalize_sequence(mode, len(axes))
        if False:
            # TODO: enable branch once in-place output is supported
            derivative(
                arr,
                axes[0],
                output,
                modes[0],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
        else:
            output = derivative(
                arr,
                axes[0],
                None,
                modes[0],
                cval,
                *extra_arguments,
                **extra_keywords,
            )

        xp.multiply(output, output, output)
        for ii in range(1, len(axes)):
            if False:
                # TODO: enable branch once in-place output is supported
                tmp = derivative(
                    arr,
                    axes[ii],
                    output.dtype,
                    modes[ii],
                    cval,
                    *extra_arguments,
                    **extra_keywords,
                )
            else:
                tmp = derivative(
                    arr,
                    axes[ii],
                    None,
                    modes[ii],
                    cval,
                    *extra_arguments,
                    **extra_keywords,
                )
            xp.multiply(tmp, tmp, tmp)
            output += tmp
        # This allows the sqrt to work with a different default casting
        xp.sqrt(output, output, casting="unsafe")
    else:
        output[...] = arr[...]
    return output


def gaussian_gradient_magnitude(
    arr, sigma, output=None, mode="reflect", cval=0.0, *, xp=None, **kwargs
):
    """Multidimensional gradient magnitude using Gaussian derivatives.

    See ``scipy.ndimage.gaussian_gradient_magnitude``

    This version supports only ``np.float32``, ``np.float64``,
    ``np.complex64`` and ``np.complex128`` dtypes.
    """
    xp, _ = get_array_module(arr, xp)
    arr = check_device(arr, xp)

    def derivative(arr, axis, output, mode, cval, sigma, *, xp=xp, **kwargs):
        order = [0] * arr.ndim
        order[axis] = 1
        return gaussian_filter(
            arr, sigma, order, output, mode, cval, xp=xp, **kwargs
        )

    return generic_gradient_magnitude(
        arr,
        derivative,
        output,
        mode,
        cval,
        extra_arguments=(sigma,),
        extra_keywords=kwargs,
        xp=xp,
    )
