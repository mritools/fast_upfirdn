from itertools import product

import numpy as np
import pytest

from scipy.signal import upfirdn as upfirdn_scipy
import fast_upfirdn
from fast_upfirdn import upfirdn

cupy = pytest.importorskip("cupy")
testing = pytest.importorskip("cupy.testing")


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
        upfirdn_scipy(h_cpu, x_cpu, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    testing.assert_allclose(
        upfirdn_scipy(h_cpu, x_cpu, up=2, down=1), upfirdn(h, x, up=2, down=1)
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
        upfirdn_scipy(h_cpu, x_cpu, up=1, down=2), upfirdn(h, x, up=1, down=2)
    )

    # down=1 kernel case
    testing.assert_allclose(
        upfirdn_scipy(h_cpu, x_cpu, up=2, down=1), upfirdn(h, x, up=2, down=1)
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
        upfirdn_scipy(h_cpu, x_cpu, up=1, down=down),
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
        upfirdn_scipy(h_cpu, x_cpu, up=up, down=1), upfirdn(h, x, up=up, down=1)
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
            upfirdn_scipy(h_cpu, x_cpu, up=1, down=2, axis=axis),
            upfirdn(h, x, up=1, down=2, axis=axis),
        )

        # down=1 case
        testing.assert_allclose(
            upfirdn_scipy(h_cpu, x_cpu, up=2, down=1, axis=axis),
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
        upfirdn_scipy(h_cpu, x_cpu, up=up, down=down),
        upfirdn(h, x, up=up, down=down),
    )
