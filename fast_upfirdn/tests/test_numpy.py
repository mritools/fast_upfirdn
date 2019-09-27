from itertools import product

import numpy as np

from fast_upfirdn import convolve, correlate
from fast_upfirdn._util import have_cupy
import pytest

array_modules = [np]
if have_cupy:
    import cupy

    array_modules += [cupy]


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, function, xp",
    product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 4, 5, 6, 7, 8],
        ["full", "valid", "same"],
        ["correlate", "convolve"],
        array_modules,
    ),
)
def test_convolve_and_correlate(dtype_x, dtype_h, len_x, mode, function, xp):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)

        if function == "convolve":
            func_cpu = np.convolve
            func_gpu = convolve
        elif function == "correlate":
            func_cpu = np.correlate
            func_gpu = correlate

        y = func_cpu(x_cpu, h_cpu, mode=mode)

        y2 = func_gpu(xp.asarray(x_cpu), xp.asarray(h_cpu), mode=mode)
        xp.testing.assert_allclose(y, y2)


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, function, xp",
    product(
        [np.float32, np.complex64],
        [np.float32, np.complex64],
        [2, 3, 4, 5, 6, 7, 8],
        ["full", "valid", "same"],
        ["correlate", "convolve"],
        array_modules,
    ),
)
def test_convolve_and_correlate_complex(
    dtype_x, dtype_h, len_x, mode, function, xp
):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    if x_cpu.dtype.kind == "c":
        x_cpu = x_cpu + 1j * x_cpu

    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        if h_cpu.dtype.kind == "c":
            h_cpu = h_cpu + 1j * h_cpu

        if function == "convolve":
            func_cpu = np.convolve
            func_gpu = convolve
        elif function == "correlate":
            func_cpu = np.correlate
            func_gpu = correlate

        y = func_cpu(x_cpu, h_cpu, mode=mode)

        y2 = func_gpu(xp.asarray(x_cpu), xp.asarray(h_cpu), mode=mode)
        xp.testing.assert_allclose(y, y2)
