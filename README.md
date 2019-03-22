**(experimental) upsampled and downsampled convolutions on the GPU**

This library currently provides three main functions:

1.) **upfirdn**  (similar to ``scipy.signal.upfirdn``, but currently requires either ``up=1`` or ``down=1``)

2.) **convolve1d**  (similar to ``scipy.ndimage.convolve1d``)

3.) **convolve_separable**  (similar to ``scipy.ndimage.convolve``, but the kernel must made of 1D kernel along each axis (i.e. separable))

convolve_separable can be used to implement the equivalent of:
``scipy.ndimage.uniform_filter``
``scipy.ndimage.gaussian_filter``

Requires:
    
- NumPy
- CuPy  (>=6.0.0a1 or so)
- SciPy (>=0.19)

Optional:
    
- pytest

**Example**

```Python

import numpy as np
import cupy
from upfirdn_gpu import convolve_separable

# separable 5x5x5 convolution kernel on the CPU
x = np.random.randn(256, 256, 256).astype(np.float32)
w = np.ones((5, ), dtype=np.float32)
convolve_separable(x, [w, ]*x.ndim)
# %timeit convolve_separable(x, [w, ]*x.ndim) -> 497 ms ± 6.42 ms

# separable 5x5x5 convolution kernel on the GPU
xg = cupy.asarray(x)
wg = cupy.asarray(w)
convolve_separable(xg, [wg, ]*x.ndim)
# %timeit convolve_separable(xg, [wg, ]*x.ndim) -> 21.2 ms ± 5.33 µs
```
