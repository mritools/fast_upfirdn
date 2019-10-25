from itertools import product
from math import gcd

import numpy as np
import pytest

from fast_upfirdn import upfirdn, resample_poly, correlate1d
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
for mode in _upfirdn_modes:
    x = cupy.array([1, 2, 3, 1], dtype=float)
    npre, npost = 6, 6
    # use impulse response filter to probe values extending past the original
    # array boundaries
    h = cupy.zeros((npre + 1 + npost, ), dtype=float)
    h[npre] = 1

    if mode == 'constant':
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
    elif mode == "constant":
        y_expected = cupy.pad(x, (npre, npost), mode=mode,
                              constant_values=cval)
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


class TestResample(object):
    """Note: The tests in this class are adapted from scipy.signal's tests"""

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_mutable_window(self, padtype):
        # Test that a mutable window is not modified
        impulse = cupy.zeros(3)
        window = cupy.random.RandomState(0).randn(2)
        window_orig = window.copy()
        resample_poly(impulse, 5, 1, window=window, padtype=padtype)

        cupy.testing.assert_array_equal(window, window_orig)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_output_float32(self, padtype):
        # Test that float32 inputs yield a float32 output
        x = cupy.arange(10, dtype=np.float32)
        h = cupy.array([1, 1, 1], dtype=np.float32)
        y = resample_poly(x, 1, 2, window=h, padtype=padtype)
        assert(y.dtype == np.float32)

    @pytest.mark.parametrize(
        "ext, padtype",
        product([False, True], padtype_options),
    )
    def test_resample_methods_on_sinusoids(self, ext, padtype):
        hann = signal.windows.hann

        # Test resampling of sinusoids
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]

        # Sinusoids, windowed to avoid edge artifacts
        t = np.arange(rate) / float(rate)
        freqs = np.array((1., 10., 40.))[:, np.newaxis]
        x = np.sin(2 * np.pi * freqs * t) * hann(rate)

        for rate_to in rates_to:
            t_to = np.arange(rate_to) / float(rate_to)
            y_tos = np.sin(2 * np.pi * freqs * t_to) * hann(rate_to)
            if ext and rate_to != rate:
                # Match default window design
                g = gcd(rate_to, rate)
                up = rate_to // g
                down = rate // g
                max_rate = max(up, down)
                f_c = 1. / max_rate
                half_len = 10 * max_rate
                window = signal.firwin(2 * half_len + 1, f_c,
                                       window=('kaiser', 5.0))
                window = cupy.asarray(window)
                polyargs = {'window': window, 'padtype': padtype}
            else:
                polyargs = {'padtype': padtype}

            x = cupy.asarray(x)
            y_resamps = resample_poly(x, rate_to, rate, axis=-1, **polyargs)
            y_resamps = y_resamps.get()

            for y_to, y_resamp, freq in zip(y_tos, y_resamps, freqs):
                if freq >= 0.5 * rate_to:
                    y_to.fill(0.)  # mostly low-passed away
                    if padtype in ['minimum', 'maximum']:
                        np.testing.assert_allclose(y_resamp, y_to, atol=3e-1)
                    else:
                        np.testing.assert_allclose(y_resamp, y_to, atol=1e-3)
                else:
                    np.testing.assert_array_equal(y_to.shape, y_resamp.shape)
                    corr = np.corrcoef(y_to, y_resamp)[0, 1]
                    assert corr > 0.99

    @pytest.mark.parametrize(
        "ext, padtype",
        product([False, True], padtype_options),
    )
    def test_resample_methods_random_input(self, ext, padtype):
        hann = signal.windows.hann

        # Test resampling of sinusoids
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]

        # Sinusoids, windowed to avoid edge artifacts
        t = np.arange(rate) / float(rate)

        # Random data
        rng = np.random.RandomState(0)
        x = hann(rate) * np.cumsum(rng.randn(rate))  # low-pass, wind
        for rate_to in rates_to:
            # random data
            t_to = np.arange(rate_to) / float(rate_to)
            y_to = np.interp(t_to, t, x)
            y_resamp_scipy = signal.resample_poly(
                x, rate_to, rate, padtype=padtype
            )
            y_resamp = resample_poly(
                cupy.asarray(x), rate_to, rate, padtype=padtype
            ).get()
            np.testing.assert_array_almost_equal(y_resamp_scipy, y_resamp)
            corr = np.corrcoef(y_to, y_resamp)[0, 1]
            assert corr > 0.99

    def test_poly_vs_filtfilt(self):
        # Check that up=1.0 gives same answer as filtfilt + slicing
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            if x.real.dtype.char == 'f':
                atol = rtol = 1e-6  # Note: scipy's tests use 1e-7
            else:
                atol = rtol = 1e-13

            # resample_poly assumes zeros outside of signl, whereas filtfilt
            # can only constant-pad. Make them equivalent:
            x[0] = 0
            x[-1] = 0

            for down in down_factors:
                h = signal.firwin(31, 1. / down, window='hamming')
                yf = signal.filtfilt(h, 1.0, x, padtype='constant')[::down]

                # Need to pass convolved version of filter to resample_poly,
                # since filtfilt does forward and backward, but resample_poly
                # only goes forward
                hc = signal.convolve(h, h[::-1])
                x_gpu, hc = cupy.asarray(x), cupy.asarray(hc)
                y = resample_poly(x_gpu, 1, down, window=hc).get()
                np.testing.assert_allclose(yf, y, atol=atol, rtol=rtol)

    def test_correlate1d(self):
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = np.random.random((nx,))
                    weights = np.random.random((nweights,))
                    x, weights = cupy.asarray(x), cupy.asarray(weights)
                    y_g = correlate1d(
                        x, weights[::-1], mode='constant'
                    )
                    y_s = resample_poly(
                        x, up=1, down=down, window=weights)
                    cupy.testing.assert_allclose(y_g[::down], y_s)
