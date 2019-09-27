from itertools import product

import numpy as np
from scipy import ndimage as ndi

from fast_upfirdn import convolve1d, correlate1d, upfirdn
from fast_upfirdn._scipy_ndimage import _get_ndimage_mode_kwargs
from fast_upfirdn._util import have_cupy

import pytest

array_modules = [np]
if have_cupy:
    import cupy

    array_modules += [cupy]


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, xp",
    product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 6, 7],
        ["constant", "mirror", "nearest", "reflect", "wrap"],
        array_modules,
    ),
)
def test_convolve1d(dtype_x, dtype_h, len_x, mode, xp):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        min_origin = -(len_h // 2)
        max_origin = (len_h - 1) // 2
        for origin in range(min_origin, max_origin + 1):
            y = ndi.convolve1d(x_cpu, h_cpu, mode=mode, cval=0, origin=origin)

            # test via convolve1d
            y3 = convolve1d(
                xp.asarray(x_cpu),
                xp.asarray(h_cpu),
                mode=mode,
                cval=0,
                origin=origin,
            )
            xp.testing.assert_allclose(y, y3)

            # test using upfirdn directly
            offset = len(h_cpu) // 2 + origin
            mode_kwargs = _get_ndimage_mode_kwargs(mode, cval=0)
            y2 = upfirdn(
                xp.asarray(h_cpu),
                xp.asarray(x_cpu),
                offset=offset,
                **mode_kwargs
            )[:len_x]
            xp.testing.assert_allclose(y, y2)

        for origin in [min_origin - 1, max_origin + 1]:
            with pytest.raises(ValueError):
                convolve1d(
                    xp.asarray(x_cpu),
                    xp.asarray(h_cpu),
                    mode=mode,
                    cval=0,
                    origin=origin,
                )


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, xp",
    product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 6, 7],
        ["constant", "mirror", "nearest", "reflect", "wrap"],
        array_modules,
    ),
)
def test_correlate1d(dtype_x, dtype_h, len_x, mode, xp):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, 2 * len_x + 2):  # include cases for len_h > len_x
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        min_origin = -(len_h // 2)
        max_origin = (len_h - 1) // 2

        for origin in range(min_origin, max_origin + 1):
            y = ndi.correlate1d(x_cpu, h_cpu, mode=mode, cval=0, origin=origin)

            # test via convolve1d
            y3 = correlate1d(
                xp.asarray(x_cpu),
                xp.asarray(h_cpu),
                mode=mode,
                cval=0,
                origin=origin,
            )
            xp.testing.assert_allclose(y, y3)

        for origin in [min_origin - 1, max_origin + 1]:
            with pytest.raises(ValueError):
                correlate1d(
                    xp.asarray(x_cpu),
                    xp.asarray(h_cpu),
                    mode=mode,
                    cval=0,
                    origin=origin,
                )


# TODO: add tests for uniform_filter and gaussian_filter
