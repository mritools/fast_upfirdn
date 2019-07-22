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
