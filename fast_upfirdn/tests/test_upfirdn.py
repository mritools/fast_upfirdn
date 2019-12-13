from itertools import product

import numpy as np
import pytest

from fast_upfirdn import upfirdn
from fast_upfirdn.cpu._upfirdn import _upfirdn_modes

cupy = pytest.importorskip("cupy")
signal = pytest.importorskip("scipy.signal")
ndimage = pytest.importorskip("scipy.ndimage")
testing = pytest.importorskip("cupy.testing")

padtype_options = ["constant", "mean", "minimum", "maximum", "line"]
# TOOD: add median once cupy.median is implemented


@pytest.mark.parametrize("mode", _upfirdn_modes)
def test_extension_modes_via_convolve(mode):
    """Test vs. manually computed results for modes not in numpy's pad."""
    x = cupy.array([1, 2, 3, 1], dtype=float)
    npre, npost = 6, 6
    # use impulse response filter to probe values extending past the original
    # array boundaries
    h = cupy.zeros((npre + 1 + npost,), dtype=float)
    h[npre] = 1

    if mode == "constant":
        cval = 5.0
        y = upfirdn(h, x, up=1, down=1, mode=mode, cval=cval)
    else:
        y = upfirdn(h, x, up=1, down=1, mode=mode)

    if mode == "antisymmetric":
        y_expected = cupy.asarray(
            [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2]
        )
    elif mode == "antireflect":
        y_expected = cupy.asarray(
            [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1]
        )
    elif mode == "smooth":
        y_expected = cupy.asarray(
            [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11]
        )
    elif mode == "line":
        lin_slope = (x[-1] - x[0]) / (len(x) - 1)
        left = x[0] + cupy.arange(-npre, 0, 1) * lin_slope
        right = x[-1] + cupy.arange(1, npost + 1) * lin_slope
        y_expected = cupy.concatenate((left, x, right))
    elif mode == "constant":
        y_expected = cupy.pad(x, (npre, npost), mode=mode, constant_values=cval)
    else:
        y_expected = cupy.pad(x, (npre, npost), mode=mode)
    y_expected = y_expected.astype(y.dtype)
    cupy.testing.assert_allclose(y, y_expected)


@pytest.mark.parametrize(
    "dtype_data, dtype_filter",
    product(
        [
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
            np.float16,
            np.uint8,
            np.int16,
            np.int32,
            int,
        ],
        [np.float32, np.float64, np.complex64, np.complex128],
    ),
)
def test_dtype_combos(dtype_data, dtype_filter):
    shape = (64, 64)
    size = int(np.prod(shape))
    x = cupy.arange(size, dtype=dtype_data).reshape(shape)
    x_cpu = x.get()
    h_cpu = np.arange(5, dtype=dtype_filter)
    h = cupy.asarray(h_cpu)

    # up=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=2, down=1), upfirdn(h, x, up=2, down=1)
    )


@pytest.mark.parametrize(
    "nh, nx", product([2, 3, 4, 5, 6, 7, 8], [16, 17, 18, 19, 20])
)
def test_input_and_filter_sizes(nh, nx):
    dtype_data = dtype_filter = np.float32
    x = cupy.arange(nx, dtype=dtype_data)
    x_cpu = x.get()
    h_cpu = np.arange(1, nh + 1, dtype=dtype_filter)
    h = cupy.asarray(h_cpu)

    # up=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=2, down=1), upfirdn(h, x, up=2, down=1)
    )


@pytest.mark.parametrize("down", [1, 2, 3, 4, 5, 6, 7, 8])
def test_down(down):
    dtype_data = dtype_filter = np.float32
    nx = 16
    nh = 4
    x = cupy.arange(nx, dtype=dtype_data)
    x_cpu = x.get()
    h_cpu = np.arange(1, nh + 1, dtype=dtype_filter)
    h = cupy.asarray(h_cpu)

    # up=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=1, down=down),
        upfirdn(h, x, up=1, down=down),
    )


@pytest.mark.parametrize("up", [1, 2, 3, 4, 5, 6, 7, 8])
def test_up(up):
    dtype_data = dtype_filter = np.float32
    nx = 16
    nh = 4
    x = cupy.arange(nx, dtype=dtype_data)
    x_cpu = x.get()
    h_cpu = np.arange(1, nh + 1, dtype=dtype_filter)
    h = cupy.asarray(h_cpu)

    # up=1 kernel case
    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=up, down=1),
        upfirdn(h, x, up=up, down=1),
    )


@pytest.mark.parametrize(
    "shape, axis, order",
    product(
        [(16, 8), (24, 16, 8), (8, 9, 10, 11)],
        [0, 1, 2, 3, -1],  # 0, 1, -1],
        ["C", "F"],
    ),
)
def test_axis_and_order(shape, axis, order):
    dtype_data = dtype_filter = np.float32
    size = int(np.prod(shape))
    x_cpu = np.arange(size, dtype=dtype_data).reshape(shape, order=order)
    h_cpu = np.arange(3, dtype=dtype_filter)
    x = cupy.asarray(x_cpu, order=order)
    h = cupy.asarray(h_cpu)
    ndim = len(shape)
    if axis >= -ndim and axis < ndim:
        # up=1 case
        testing.assert_allclose(
            signal.upfirdn(h_cpu, x_cpu, up=1, down=2, axis=axis),
            upfirdn(h, x, up=1, down=2, axis=axis),
        )

        # down=1 case
        testing.assert_allclose(
            signal.upfirdn(h_cpu, x_cpu, up=2, down=1, axis=axis),
            upfirdn(h, x, up=2, down=1, axis=axis),
        )
    else:
        with pytest.raises(ValueError):
            upfirdn(h, x, up=2, down=1, axis=axis)


@pytest.mark.parametrize(
    "up, down, nx, nh",
    product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [16, 17], [3, 4]),
)
def test_general_up_and_down(up, down, nx, nh):
    dtype_data = dtype_filter = np.float32
    x = cupy.arange(nx, dtype=dtype_data)
    x_cpu = x.get()
    h_cpu = np.arange(1, nh + 1, dtype=dtype_filter)
    h = cupy.asarray(h_cpu)

    testing.assert_allclose(
        signal.upfirdn(h_cpu, x_cpu, up=up, down=down),
        upfirdn(h, x, up=up, down=down),
    )
