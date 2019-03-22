import numpy as np
import cupy


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
        if isinstance(arr, np.ndarray):
            return np, False
        else:
            xp = cupy.get_array_module(arr)
            return xp, (xp != np)
    else:
        return xp, (xp != np)
