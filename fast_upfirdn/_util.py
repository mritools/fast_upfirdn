"""Miscellaneous utility functions."""

import numpy as np

try:
    import cupy

    have_cupy = True
except ImportError:
    have_cupy = False

allow_device_transfers = False

"""
@profile decorator that does nothing when line_profiler is not active

see:
http://stackoverflow.com/questions/18229628/python-profiling-using-line-profiler-clever-way-to-remove-profile-statements
"""
try:
    import builtins

    profile = builtins.__dict__["profile"]
except (AttributeError, KeyError):

    def profile(func):
        """No line profiler, provide a pass-through version."""
        return func


__all__ = [
    "allow_device_transfers",
    "have_cupy",
    "get_array_module",
    "array_on_device",
    "check_device",
]


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


def check_device(arr, xp, autoconvert=allow_device_transfers):
    if autoconvert:
        return array_on_device(arr, xp)
    elif xp == np:
        if isinstance(arr, np.ndarray):
            return arr
        elif hasattr(arr, "__array_interface__") and not (
            have_cupy and hasattr(arr, "__cuda_array_interface__")
        ):
            return np.asarray(arr)
        else:
            raise ValueError(
                "Expected a numpy.ndarray, got {}".format(type(arr))
            )
    elif have_cupy and xp == cupy:
        if isinstance(arr, cupy.ndarray):
            return arr
        elif hasattr(arr, "__cuda_array_interface__"):
            return cupy.asarray(arr)
        else:
            raise ValueError(
                "Expected a cupy.ndarray, got {}".format(type(arr))
            )
    else:
        raise ValueError("unsupported module: {}".format(xp))


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
        if have_cupy and hasattr(arr, "__cuda_array_interface__"):
            # copy back from GPU
            return arr.get()
    return xp.asarray(arr)
