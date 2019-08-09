import numpy as np

try:
    import cupy
    have_cupy = True
except ImportError:
    have_cupy = False

__all__ = ['have_cupy', 'get_array_module', 'array_on_device']


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
    """Transfer arr to device corresponding to xp.

    Paramters
    ---------
    arr : np.ndarray or cupy.ndarray
        The array.
    xp : {numpy, cupy}
        The desired ndarray module.

    Returns
    -------
    A cupy.ndarray if ``xp == cupy``, otherwise a numpy.ndarray.
    """
    if xp == np:
        if have_cupy and hasattr(arr, '__cuda_array_interface__'):
            # copy back from GPU
            return arr.get()
    return xp.asarray(arr)


# arr, weights = _fixup_dtypes(arr, weights)
def _fixup_dtypes(data, h):
    """Converts data and h to the nearest supported floating point type.

    Parameters
    ----------
    data, h : ndarray
        Input arrays.

    Returns
    -------
    data, h : ndarray
        Arrays converted to the nearest common dtype supported by the library.
    dtype_out : np.dtype
        The dtype.

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
