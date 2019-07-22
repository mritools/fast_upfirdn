import numpy as np
import scipy.ndimage as ndi

import cupy

from fast_upfirdn._util import get_array_module, array_on_device, have_cupy
if have_cupy:
    from fast_upfirdn.cupy._upfirdn import (
        _convolve1d as _convolve1d_gpu,
        _nearest_supported_float_dtype,
    )


__all__ = [
    "convolve1d",
    "correlate1d",
    "uniform_filter1d",
    "uniform_filter",
    "convolve_separable",
]


def _check_axis(axis, rank):
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("invalid axis")
    return axis


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if output is None:
        output = cupy.zeros(shape, dtype=input.dtype.name)
    elif type(output) in [type(type), type(cupy.zeros((4,)).dtype)]:
        output = cupy.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = cupy.typeDict[output]
        output = cupy.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


# input, weights = _fixup_dtypes(input, weights)
def _fixup_dtypes(data, h):
    """Compile an upfirdn kernel based on dtype.

    Also converts h, data to the nearest supported floating point type.
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


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    if hasattr(input, "__iter__") and not isinstance(input, str):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized


def convolve1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    xp=None,
    crop=True,
):
    """Calculate a one-dimensional convolution along the given axis.

    The lines of the array along the given axis are convolved with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : ndarray
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s

    Returns
    -------
    convolve1d : ndarray
        Convolved array with same shape as input

    Examples
    --------
    >>> from scipy.ndimage import convolve1d
    >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([14, 24,  4, 13, 12, 36, 27,  0])
    """
    xp = cupy.get_array_module(input)
    if xp == np:
        if crop != True:
            raise ValueError(
                "ndimage.convolve1d only supports cropped convolution.")
        return ndi.convolve1d(
            input,
            weights,
            axis=axis,
            output=output,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    axis = _check_axis(axis, input.ndim)

    # input, weights, dtype_out = _fixup_dtypes(input, weights)
    # if output is None:
    #     output = cupy.zeros(input.shape, dtype=dtype_out)
    if output is not None:
        raise NotImplementedError("in-place operation not implemented")

    w_len_half = len(weights) // 2
    if input.shape[axis] < w_len_half:
        raise ValueError("input.shape[axis] < h.size//2 unsupported")
    if origin < -w_len_half or origin > ((len(weights) - 1) // 2):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= (len(weights)-1) // 2"
        )
    if mode == "reflect":
        mode_kwarg = dict(mode="symmetric")
    elif mode == "mirror":
        mode_kwarg = dict(mode="reflect")
    elif mode == "nearest":
        mode_kwarg = dict(mode="constant")
    elif mode == "constant":
        if cval == 0:
            mode_kwarg = dict(mode="zero")
        else:
            raise NotImplementedError("mode 'constant' not implemented")
    elif mode == "wrap":
        mode_kwarg = dict(mode="periodic")
    else:
        raise ValueError("unsupported mode: {}".format(mode))
    tmp = _convolve1d_gpu(
        weights,
        input,
        axis=axis,
        origin=origin,
        crop=crop,
        out=output,
        **mode_kwarg
    )
    return tmp
    # out_slices = [slice(None), ] * input.ndim
    # nedge = w_len_half
    # out_slices[axis] = slice(nedge, nedge + input.shape[axis])
    # return tmp[tuple(out_slices)]


def correlate1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    xp=None,
):
    """Calculate a one-dimensional correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return convolve1d(
        input,
        weights,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        xp=xp,
    )


def uniform_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Calculate a one-dimensional uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a
    uniform filter of given size.

    Parameters
    ----------
    %(input)s
    size : int
        length of uniform filter
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s

    Examples
    --------
    >>> from scipy.ndimage import uniform_filter1d
    >>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([4, 3, 4, 1, 4, 6, 6, 3])
    """
    xp = cupy.get_array_module(input)
    input = xp.asarray(input)
    # axis = _check_axis(axis, input.ndim)
    if size < 1:
        raise RuntimeError("incorrect filter size")
    # output = _get_output(output, input)  # TODO: add output support
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError("invalid origin")
    weights = xp.full(
        (size,), 1 / size, dtype=np.result_type(input.real.dtype, np.float32)
    )
    return convolve1d(input, weights, axis, output, mode, cval, origin, xp=xp)


def uniform_filter(
    input, size=3, output=None, mode="reflect", cval=0.0, origin=0
):
    """Multi-dimensional uniform filter.

    Parameters
    ----------
    %(input)s
    size : int or sequence of ints, optional
        The sizes of the uniform filter are given for each axis as a
        sequence, or as a single number, in which case the size is
        equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s

    Returns
    -------
    uniform_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional uniform filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = ndimage.uniform_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    xp = cupy.get_array_module(input)
    # input = xp.asarray(input)
    # # output = _ni_support._get_output(output, input)  # TODO
    # dtype_input = _nearest_supported_float_dtype(input.dtype)
    # if input.dtype != dtype_input:
    #     input = input.astype(dtype_input)
    # if output is None:
    #     output = cupy.zeros(input.shape, dtype=dtype_input)

    sizes = _normalize_sequence(size, input.ndim)
    origins = _normalize_sequence(origin, input.ndim)
    modes = _normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [
        (axes[ii], sizes[ii], origins[ii], modes[ii])
        for ii in range(len(axes))
        if sizes[ii] > 1
    ]
    if len(axes) > 0:

        # TODO
        # for axis, size, origin, mode in axes:
        #     uniform_filter1d(input, int(size), axis, output, mode,
        #                      cval, origin)
        #     input = output

        for axis, size, origin, mode in axes:
            input = uniform_filter1d(
                input, int(size), axis, output, mode, cval, origin
            )
        output = input
    else:
        # output[...] = input[...]
        output = input.copy()  # TODO
    return output


def convolve_separable(x, w, xp=None, **kwargs):
    """n-dimensional convolution via separable application of convolve1d

    currently a single 1d filter, w, is applied on all axes.
    w can also be a list of filters (equal in length to the number of axes)
    """
    axes = kwargs.pop("axes", range(x.ndim))
    xp = cupy.get_array_module(x)
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


"""
equivalent of numpy.convolve
"""


def convolve(a, v, mode="full"):
    a = cupy.array(a, copy=False, ndmin=1)
    v = cupy.array(v, copy=False, ndmin=1)

    # upgrade a, v to a common dtype
    output_dtype = np.result_type(a.dtype, v.dtype)
    a = a.astype(output_dtype, copy=False)
    v = v.astype(output_dtype, copy=False)

    if len(v) > len(a):
        a, v = v, a
    if len(a) == 0:
        raise ValueError("a connot be empty")
    if len(v) == 0:
        raise ValueError("v connot be empty")
    if a.ndim != v.ndim != 1:
        raise ValueError("convolve only supports 1D arrays")
    origin = 0
    if mode == "full":
        crop = False
    elif mode in ["same", "valid"]:
        crop = True
        if len(v) > 2 and len(v) % 2 == 0:
            origin = -1
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    # Note _convolve1d_gpu always computes in floating point
    out = _convolve1d_gpu(
        v, a, axis=0, origin=origin, crop=crop, mode="zero"
    )

    if mode == "valid":
        sl_start = len(v) // 2
        sl_end = sl_start + len(a) - len(v) + 1
        out = out[sl_start:sl_end]

    # restore expected NumPy output dtype
    out = out.astype(output_dtype, copy=False)
    return out


def correlate(a, v, mode="valid"):
    v = v[::-1]
    if cupy.iscomplexobj(v):
        cupy.conj(v)
    return convolve(a, v, mode=mode)
