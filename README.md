# separable n-dimensional convolution on the CPU and GPU

The core low-level function implemented here is an equivalent of
[scipy.signal.upfirdn] but with support for both CPU (via [NumPy]) and GPU
(via [CuPy]). It can be installed without SciPy itself.

This package can still be installed without CuPy, but only the CPU-based
implementation will be avialable.

This function can be used to implement a wide variety of separable
convolution-based filtering operations on n-dimensional arrays.

The version of ``upfirdn`` here supports several signal extension modes. These
have been contributed upstream to SciPy and are available there for SciPy 1.4+.
A few additional keyword-only arguments are present in the ``upfirdn``
implementation here that do not exist in SciPy. These should be considered
experimental and subject to change.

**Requires:**

- [NumPy]  (>=1.14)
- [Cython]  (>=0.29.13)  (needed during build, not at runtime)

**Recommended:**

- [CuPy]  (>=6.1)

**To run the test suite, users will also need:**

- [pytest]
- [SciPy]  (>=1.0.1)


See ``requirements-dev.txt`` for any additional requirements needed for
development.

**Installation:**

This package is in the early stages of development and does not yet have
binary wheels. Source packages are available on PyPI.

```
pip install fast_upfirdn
```

Developers can download the source from GitHub, navigate to the source
directory and run:

```
python -m pip install -e . -v  --no-build-isolation --no-use-pep517
```

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

On the GPU there is also a faster code path for the case ``up=1, down=1`` that
can be called via ``fast_upfirdn.cupy.convolve1d``
```Python
fast_upfirdn.cupy.convolve1d(x_d, h_d)
```

## Similar Software

The [RAPIDS] project [cuSignal] provides a more comprehensive implementation
of functions from ``scipy.signal``. Like ``fast_upfirdn``, it also depends on
[CuPy], but has an additional dependency on [Numba].

One advantage of this repository is that it supports the new boundary handling
modes introduced for ``upfirdn`` in SciPy 1.4, while at the time of writing
(Jan 2019), [cuSignal] does not.


[conda]: https://docs.conda.io/en/latest/
[CuPy]: https://cupy.chainer.org
[cuSignal]: https://github.com/rapidsai/cusignal
[Cython]: https://cython.org/
[Numba]: numba.pydata.org
[NumPy]: https://numpy.org/
[PyPI]: https://pypi.org
[pytest]: https://docs.pytest.org/en/latest/
[RAPIDS]: https://rapids.ai
[SciPy]: https://scipy.org
[scikit-image]: https://scikit-image.org
[scipy.signal.upfirdn]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html
