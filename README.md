# separable n-dimensional convolution using CuPy

The core low-level function implemented here is an equivalent of
``scipy.signal.upfirdn`` but with support for both CPU (via [NumPy]) and GPU
(via [CuPy]).

This function can be used to implement a wide variety of separable
convolution-based filtering operations on n-dimensional arrays. A subset of
functions from ``scipy.signal``, ``scipy.ndimage`` and ``scikit-image`` built
on top of fast-upfirdn are provided in the [cupyext] library.

The version of ``upfirdn`` here supports several signal extension modes. These
have been contributed upstream to SciPy and are available there for SciPy 1.4+.
A few additional keyword-only arguments are present in the present
implementation that do not exist in SciPy. These should be considered
experimental and subject to change.

**Requires:**

- [NumPy]
- [CuPy]  (>=6.1)
- [SciPy]  (>=0.19)

Optional (for testing/development):

- [pytest]

**Installation:**

This package is in the early stages of development and is not yet available via
[PyPI] or [conda]. Users can download the source from GitHub, navigate to
the source directory and run:

`Python
pip install . -v
`

**Usage:**


The primary function provided by this package is `upfirdn`:

The top-level ``upfirdn`` autoselects CPU or GPU (CUDA) execution based on
whether the input data was a NumPy or CuPy array.


The following will run on the CPU because the inputs are NumPy arrays
```Python
import numpy as np
import fast_upfirdn

x = np.arange(8)
h = np.ones(3)

fast_upfirdn.upfirdn(x, h, up=1, down=2)
```

The following will run on the GPU because the inputs are CuPy arrays
```Python
import cupy as cp
x_d = cp.arange(8)
h_d = cp.ones(3)
fast_upfirdn.upfirdn(x_d, h_d, up=1, down=2)
```

Alternatively the CPU version can be called directly as
``fast_upfirdn.cpu.upfirdn``

```Python
fast_upfirdn.cpu.upfirdn(x, h, up=1, down=2)
```

Or the GPU version can be called specifically as ``fast_upfirdn.cupy.upfirdn``
```Python
fast_upfirdn.cupy.upfirdn(x_d, h_d, up=1, down=2)
```

On the GPU there is also a faster code path for the case up=1, down=1 that
can be called via
```Python
fast_upfirdn.cupy.convolve1d(x_d, h_d)
```

## Similar Software

The [RAPIDS] project [cuSignal] provides a more comprehensive implementation
of functions from scipy.signal. Like [cupyext], it also depends on [CuPy], but
has an additional dependency on [Numba]. One other difference is at the time of
writing, it does not support all of the new boundary handling modes introduced
in SciPy 1.4.

[conda]: https://docs.conda.io/en/latest/
[CuPy]: https://cupy.chainer.org
[cupyext]: https://github.com/grlee77/cupyext
[cuSignal]: https://github.com/rapidsai/cusignal
[Numba]: numba.pydata.org
[NumPy]: https://numpy.org/
[PyPI]: https://pypi.org
[pytest]: https://docs.pytest.org/en/latest/
[RAPIDS]: https://rapids.ai
[SciPy]: https://scipy.org
[scikit-image]: https://scikit-image.org
