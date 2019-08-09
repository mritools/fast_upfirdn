from itertools import product

import numpy as np
from scipy import ndimage as ndi

from fast_upfirdn import convolve, convolve1d, upfirdn
from fast_upfirdn._util import have_cupy
import fast_upfirdn
import pytest

array_modules = [np]
if have_cupy:
    import cupy
    array_modules += [cupy]


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, xp",
    product(
        [
            np.float32,
            np.float64,
        ],
        [
            np.float32,
            np.float64,
        ],
        [2, 3, 4, 5, 6, 7, 8],
        ['full', 'valid', 'same'],
        array_modules,
    ),
)
def test_convolve1d(dtype_x, dtype_h, len_x, mode, xp):
    for len_h in range(1, len_x):
        x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        for origin in range(-(len_h//2), (len_h - 1) //2):
            y = ndi.convolve1d(x_cpu, h_cpu, mode='constant', cval=0,
                               origin=origin)
            offset = len(h_cpu) // 2 + origin

            # test using upfirdn directly
            y2 = upfirdn(xp.asarray(h_cpu), xp.asarray(x_cpu), mode='constant',
                         cval=0, offset=offset)[:len_x]
            xp.testing.assert_allclose(y, y2)

            # test via convolve1d
            y3 = convolve1d(xp.asarray(x_cpu), xp.asarray(h_cpu),
                            mode='constant', cval=0, origin=origin)
            xp.testing.assert_allclose(y, y3)

# TODO: add tests for convolve1d & correlate1d with other boundary modes
#       add tests for uniform_filter and gaussian_filter
