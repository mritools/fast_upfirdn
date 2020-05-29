from itertools import product

import numpy as np
import pytest

import scipy
from scipy.signal import upfirdn as upfirdn_scipy
from fast_upfirdn import upfirdn
from fast_upfirdn.cpu._upfirdn_apply import _pad_test
from fast_upfirdn.cpu._upfirdn import _upfirdn_modes


def assert_allclose2(x, y):
    if np.lib.NumpyVersion(scipy.__version__) < np.lib.NumpyVersion("1.5.0"):
        x = x[tuple([slice(s) for s in y.shape])]
    np.testing.assert_allclose(x, y)


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
def test_dtype_combos_cpu(dtype_data, dtype_filter):
    shape = (64, 64)
    size = int(np.prod(shape))
    x = np.arange(size, dtype=dtype_data).reshape(shape)
    h = np.arange(5, dtype=dtype_filter)

    # up=1 kernel case
    assert_allclose2(
        upfirdn_scipy(h, x, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    assert_allclose2(
        upfirdn_scipy(h, x, up=2, down=1), upfirdn(h, x, up=2, down=1)
    )


@pytest.mark.parametrize(
    "nh, nx", product([2, 3, 4, 5, 6, 7, 8], [16, 17, 18, 19, 20])
)
def test_input_and_filter_sizes_cpu(nh, nx):
    dtype_data = dtype_filter = np.float32
    x = np.arange(nx, dtype=dtype_data)
    h = np.arange(1, nh + 1, dtype=dtype_filter)

    # up=1 kernel case
    assert_allclose2(
        upfirdn_scipy(h, x, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    assert_allclose2(
        upfirdn_scipy(h, x, up=2, down=1), upfirdn(h, x, up=2, down=1)
    )


@pytest.mark.parametrize(
    "shape, axis, order",
    product(
        [(16, 8), (24, 16, 8), (8, 9, 10, 11)],
        [0, 1, 2, 3, -1],  # 0, 1, -1],
        ["C", "F"],
    ),
)
def test_axis_and_order_cpu(shape, axis, order):
    dtype_data = dtype_filter = np.float32
    size = int(np.prod(shape))
    x = np.arange(size, dtype=dtype_data).reshape(shape, order=order)
    h = np.arange(3, dtype=dtype_filter)
    ndim = len(shape)
    if axis >= -ndim and axis < ndim:
        # up=1 case
        assert_allclose2(
            upfirdn_scipy(h, x, up=1, down=2, axis=axis),
            upfirdn(h, x, up=1, down=2, axis=axis),
        )

        # down=1 case
        assert_allclose2(
            upfirdn_scipy(h, x, up=2, down=1, axis=axis),
            upfirdn(h, x, up=2, down=1, axis=axis),
        )
    else:
        with pytest.raises(ValueError):
            upfirdn(h, x, up=2, down=1, axis=axis)


@pytest.mark.parametrize(
    "up, down, nx, nh",
    product([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [16, 17], [3, 4]),
)
def test_general_up_and_down_cpu(up, down, nx, nh):
    dtype_data = dtype_filter = np.float32
    x = np.arange(nx, dtype=dtype_data)
    h = np.arange(1, nh + 1, dtype=dtype_filter)

    assert_allclose2(
        upfirdn_scipy(h, x, up=up, down=down), upfirdn(h, x, up=up, down=down)
    )


@pytest.mark.parametrize("mode", _upfirdn_modes)
def test_extension_modes(mode):
    """Test vs. manually computed results for modes not in numpy's pad."""
    x = np.array([1, 2, 3, 1], dtype=float)
    npre, npost = 6, 6
    y = _pad_test(x, npre=npre, npost=npost, mode=mode)
    if mode == "antisymmetric":
        y_expected = np.asarray(
            [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2]
        )
    elif mode == "antireflect":
        y_expected = np.asarray(
            [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1]
        )
    elif mode == "smooth":
        y_expected = np.asarray(
            [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11]
        )
    elif mode == "line":
        lin_slope = (x[-1] - x[0]) / (len(x) - 1)
        left = x[0] + np.arange(-npre, 0, 1) * lin_slope
        right = x[-1] + np.arange(1, npost + 1) * lin_slope
        y_expected = np.concatenate((left, x, right))
    else:
        y_expected = np.pad(x, (npre, npost), mode=mode)
    np.testing.assert_allclose(y, y_expected)


@pytest.mark.parametrize("mode", _upfirdn_modes)
def test_extension_modes_via_convolve(mode):
    """Test vs. manually computed results for modes not in numpy's pad."""
    x = np.array([1, 2, 3, 1], dtype=float)
    npre, npost = 6, 6
    # use impulse response filter to probe values extending past the original
    # array boundaries
    h = np.zeros((npre + 1 + npost,), dtype=float)
    h[npre] = 1

    if mode == "constant":
        cval = 5.0
        y = upfirdn(h, x, up=1, down=1, mode=mode, cval=cval)
    else:
        y = upfirdn(h, x, up=1, down=1, mode=mode)

    if mode == "antisymmetric":
        y_expected = np.asarray(
            [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2]
        )
    elif mode == "antireflect":
        y_expected = np.asarray(
            [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1]
        )
    elif mode == "smooth":
        y_expected = np.asarray(
            [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11]
        )
    elif mode == "line":
        lin_slope = (x[-1] - x[0]) / (len(x) - 1)
        left = x[0] + np.arange(-npre, 0, 1) * lin_slope
        right = x[-1] + np.arange(1, npost + 1) * lin_slope
        y_expected = np.concatenate((left, x, right))
    elif mode == "constant":
        y_expected = np.pad(x, (npre, npost), mode=mode, constant_values=cval)
    else:
        y_expected = np.pad(x, (npre, npost), mode=mode)
    y_expected = y_expected.astype(y.dtype)
    np.testing.assert_allclose(y, y_expected)
