from timeit import timeit

setup = """
import cupy
import numpy as np
from scipy.signal import upfirdn as upfirdn_scipy
from upfirdn_gpu import upfirdn

d = cupy.cuda.device.Device()
order = 'C'
shape = (192, 192, 192)
down = 2
up = 1
dtype_data = dtype_filter = np.float32
size = int(np.prod(shape))
x_cpu = np.arange(size, dtype=dtype_data).reshape(shape, order=order)
h_cpu = np.arange(3, dtype=dtype_filter)
x = cupy.asarray(x_cpu, order=order)
h = cupy.asarray(h_cpu)
"""

# warm start to avoid overhead from imports and initial kernel compilation
timeit("upfirdn_scipy(h_cpu, x_cpu, up=up, down=down, axis=-1)", setup=setup, number=1)
timeit("upfirdn_scipy(h_cpu, x_cpu, up=up, down=down, axis=0)", setup=setup, number=1)
timeit("upfirdn(h, x, up=up, down=down, axis=0)", setup=setup, number=1)
timeit("upfirdn(h, x, up=up, down=down, axis=-1)", setup=setup, number=1)

nreps = 100
t_cpu_cont = timeit("upfirdn_scipy(h_cpu, x_cpu, up=up, down=down, axis=-1)",
                    setup=setup, number=nreps) / nreps
print("Duration (CPU, contiguous axis) = {} ms".format(1000 * t_cpu_cont))

t_cpu_noncont = timeit("upfirdn_scipy(h_cpu, x_cpu, up=up, down=down, axis=0)",
                       setup=setup, number=nreps) / nreps
print("Duration (CPU, non-contiguous axis) = {} ms".format(1000 * t_cpu_noncont))

t_gpu_cont = timeit("upfirdn(h, x, up=up, down=down, axis=-1); d.synchronize()",
                    setup=setup, number=nreps) / nreps
print("Duration (GPU, contiguous axis) = {} ms".format(1000 * t_gpu_cont))

t_gpu_noncont = timeit("upfirdn(h, x, up=up, down=down, axis=0); d.synchronize()",
                       setup=setup, number=nreps) / nreps
print("Duration (GPU, non-contiguous axis) = {} ms".format(1000 * t_gpu_noncont))
