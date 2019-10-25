from itertools import product

import numpy as np
import pytest

from fast_upfirdn import convolve_separable

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize(
    "shape, filter_length, dtype, mode",
    product(
        [(12,), (16, 14), (7, 5, 8)],  # shape
        [3, 4],  # filter_length
        [np.float32, np.float64, np.complex64, np.complex128],  # dtype
        ["reflect", "constant", "nearest", "mirror", "wrap"],  # mode
    ),
)
def test_convolve_separable_cpu_vs_gpu(shape, filter_length, dtype, mode):
    if dtype in [np.float32, np.complex64]:
        atol = rtol = 1e-5
    else:
        atol = rtol = 1e-10
    x = np.arange(int(np.prod(shape))).reshape(*shape).astype(dtype)
    if x.dtype.kind == "c":
        x = x + 1j * x[::-1]
    w = np.arange(1, 1 + filter_length).astype(x.real.dtype)

    # on cpu
    y = convolve_separable(x, w)

    # on gpu
    xg = cupy.asarray(x)
    wg = cupy.arange(1, 1 + filter_length)
    yg = convolve_separable(xg, wg)

    cupy.testing.assert_allclose(y, yg, atol=atol, rtol=rtol)
