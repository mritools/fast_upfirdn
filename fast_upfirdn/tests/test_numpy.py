from itertools import product

import numpy as np
from scipy import ndimage as ndi

from fast_upfirdn import convolve, convolve1d, upfirdn
from fast_upfirdn._util import have_cupy
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
def test_convolve(dtype_x, dtype_h, len_x, mode, xp):
    for len_h in range(1, len_x):
        x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)

        y = np.convolve(x_cpu, h_cpu, mode=mode)

        y2 = convolve(xp.asarray(x_cpu), xp.asarray(h_cpu), mode=mode)
        xp.testing.assert_allclose(y, y2)

# TODO: add tests for correlate
