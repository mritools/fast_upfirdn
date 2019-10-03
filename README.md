# separable n-dimensional convolution using CuPy

The core low-level function implemented here is an equivalent of
scipy.signal.upfirdn but with support for both CPU (via NumPy) and GPU
(via CuPy).

This function can be used to implement a wide variety of separable
convolution-based filtering operations on n-dimensional arrays.

As an example, a number of relevant functions from scipy and NumPy APIs have
been implemented.

These currently include:

**From scipy.signal**:

   - ``scipy.signal.upfirdn``
   - ``scipy.signal.resample_poly``

**From numpy**:

   - ``numpy.convolve`` (floating point convolutions only)
   - ``numpy.correlate`` (floating point convolutions only)

**From scipy.ndimage**:

   - ``scipy.ndimage.convolve1d``
   - ``scipy.ndimage.correlate1d``
   - ``scipy.ndimage.gaussian_filter1d``
   - ``scipy.ndimage.gaussian_filter``
   - ``scipy.ndimage.uniform_filter1d``
   - ``scipy.ndimage.uniform_filter``
   - ``scipy.ndimage.prewitt``
   - ``scipy.ndimage.sobel``
   - ``scipy.ndimage.generic_laplace``
   - ``scipy.ndimage.laplace``
   - ``scipy.ndimage.gaussian_laplace``
   - ``scipy.ndimage.generic_gradient_magnitude``
   - ``scipy.ndimage.gaussian_gradient_magnitude``

Requires:

- NumPy
- CuPy  (>=6.0.0a1 or so)
- SciPy (>=0.19)

Optional (for testing/development):

- pytest

**Example**
```Python
import numpy as np
import cupy
from fast_upfirdn import uniform_filter, convolve_separable

# separable 5x5x5 convolution kernel on the CPU
x = np.random.randn(256, 256, 256).astype(np.float32)
y = uniform_filter(x, size=5)
# %timeit convolve_separable(x, [w, ]*x.ndim)
#    -> 669 ms ± 70.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# separable 5x5x5 convolution kernel on the GPU
xg = cupy.asarray(x)
wg = cupy.asarray(w)
yg = uniform_filter(xg, size=5)
# %timeit yg = uniform_filter(xg, size=5)
#    -> 33.2 ms ± 20.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

wg = cupy.ones((5, ), dtype=np.float32)
yg = convolve_separable(xg, [wg, ] * xg.ndim)
# %timeit convolve_separable(xg, [wg, ]*x.ndim)
#    -> 20 ms ± 4.79 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
