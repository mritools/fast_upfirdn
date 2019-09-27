from math import ceil

import numpy as np

import cupy
from cupy.util import memoize
from fast_upfirdn.cpu._upfirdn_apply import mode_enum as _get_mode_enum


try:
    # Device Attributes require CuPy > 6.0.b3
    d = cupy.cuda.device.Device(0)
    cuda_MaxBlockDimX = d.attributes["MaxBlockDimX"]
    cuda_MaxGridDimX = d.attributes["MaxGridDimX"]
except (cupy.cuda.runtime.CUDARuntimeError, AttributeError):
    # guess
    cuda_MaxBlockDimX = 1024
    cuda_MaxGridDimX = 2147483647


def _output_len(len_h, in_len, up, down):
    """The output length that results from a given input.

    scipy.signal._upfirdn._output_len
    """
    in_len_copy = in_len + (len_h + (-len_h % up)) // up - 1
    nt = in_len_copy * up
    need = nt // down
    if nt % down > 0:
        need += 1
    return need


_include = r"""
#include <cupy/complex.cuh>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
    // atomicAdd for doubles didn't exist prior to compute capability 6.0
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                 (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif

// atomicAdd for complex floats via two real-valued atomicAdds

__device__ complex<float> atomicAdd(complex<float>* address,
                                    complex<float> val)
{
    float *p = reinterpret_cast<float *>(address);

    return complex<float>(atomicAdd(p, val.real()),
                          atomicAdd(p + 1, val.imag()));
}

__device__ complex<double> atomicAdd(complex<double>* address,
                                     complex<double> val)
{
    double *p = reinterpret_cast<double *>(address);

    return complex<double>(atomicAdd(p, val.real()),
                           atomicAdd(p + 1, val.imag()));
}

enum MODE {
    MODE_CONSTANT = 0,
    MODE_SYMMETRIC = 1,
    MODE_CONSTANT_EDGE = 2,
    MODE_SMOOTH = 3,
    MODE_PERIODIC = 4,
    MODE_REFLECT = 5,
    MODE_ANTISYMMETRIC = 6,
    MODE_ANTIREFLECT = 7
};


__device__
DTYPE_DATA _extend_left(DTYPE_DATA *x, long long idx, long long len_x,
                        MODE mode, DTYPE_DATA cval)
{
    DTYPE_DATA le = 0.;

    switch(mode)
    {
    // note: idx will be < 0
    case MODE_SYMMETRIC:
        if ((-idx) < len_x)
        {
            return x[-idx - 1];
        }
        else
        {
            // general case for multiple reflections:
            // the pattern repeats with periodicity 2*len_x;
            idx = (-idx - 1) % (2 * len_x);
            if (idx < len_x)
                return x[idx];
            else
                return x[len_x - 1 - (idx - len_x)];
        }
    case MODE_REFLECT:
        if ((-idx) < (len_x - 1))
        {
            return x[-idx];
        }
        else
        {
            // general case for multiple reflections:
            // the pattern repeats with periodicity 2*(len_x - 1);
            idx = (-idx - 1) % (2 * (len_x - 1));
            if (idx < (len_x - 1))
                return x[idx + 1];
            else
                return x[len_x - 2 - (idx - (len_x - 1))];
        }
    case MODE_PERIODIC:
        idx = (-idx - 1) % len_x;
        return x[len_x - idx - 1];
    case MODE_SMOOTH:
        return x[0] + (DTYPE_DATA)idx * (x[1] - x[0]);
    case MODE_ANTISYMMETRIC:
        if ((-idx) < len_x)
        {
            return -x[-idx - 1];
        }
        else
        {
            idx = (-idx - 1) % (2 * len_x);
            if (idx < len_x)
            {
                return -x[idx];
            }
            else
            {
                return x[len_x - 1 - (idx - len_x)];
            }
        }
    case MODE_ANTIREFLECT:
        if ((-idx) < len_x)
        {
            return x[0] - (x[-idx] - x[0]);
        }
        else
        {
            le = x[0] + (x[0] - x[len_x - 1]) *
                 ((DTYPE_DATA)((-(idx) - 1) / (len_x - 1)));
            idx = (-idx - 1) % (2 * (len_x - 1));
            if (idx < (len_x - 1))
            {
                return le - (x[idx + 1] - x[0]);
            }
            else
            {
                return le - (
                    x[len_x - 1] - x[len_x - 2 - (idx - (len_x - 1))]);
            }
        }
    case MODE_CONSTANT_EDGE:
        return x[0];
    case MODE_CONSTANT:
        return cval;
    default:
        return -1.;
    }
}


__device__
DTYPE_DATA _extend_right(DTYPE_DATA *x, long long idx, long long len_x,
                           MODE mode, DTYPE_DATA cval)
{
    // note: idx will be >= len_x
    DTYPE_DATA re = 0.;
    switch(mode)
    {

        case MODE_SYMMETRIC:
        {
            if (idx < (2 * len_x))
            {
                return x[len_x - 1 - (idx - len_x)];
            }
            else
            {
                idx = idx % (2 * len_x);
                if (idx < len_x)
                {
                    return x[idx];
                }
                else
                {
                    return x[len_x - 1 - (idx - len_x)];
                }
            }
        }
        case MODE_REFLECT:
        {
            if (idx < (2 * len_x - 1))
            {
                return x[len_x - 2 - (idx - len_x)];
            }
            else
            {
                idx = idx % (2 * (len_x - 1));
                if (idx < (len_x - 1))
                {
                    return x[idx];
                }
                else
                {
                    return x[len_x - 1 - (idx - (len_x - 1))];
                }
            }
        }
        case MODE_PERIODIC:
        {
            return x[idx % len_x];
        }
        case MODE_SMOOTH:
            return x[len_x - 1] +
                   (DTYPE_DATA)(idx - len_x + 1) *
                   (x[len_x - 1] - x[len_x - 2]);
        case MODE_CONSTANT_EDGE:
            return x[len_x - 1];
        case MODE_ANTISYMMETRIC:
            if (idx < (2 * len_x))
            {
                return -x[len_x - 1 - (idx - len_x)];
            }
            else
            {
                idx = idx % (2 * len_x);
                if (idx < len_x)
                {
                    return x[idx];
                }
                else
                {
                    return -x[len_x - 1 - (idx - len_x)];
                }
            }
        case MODE_ANTIREFLECT:
            if (idx < (2 * len_x - 1))
            {
                return x[len_x - 1] - (
                    x[len_x - 2 - (idx - len_x)] - x[len_x - 1]);
            }
            else
            {
                re = x[len_x - 1] +
                     (x[len_x - 1] - x[0]) *
                     ((DTYPE_DATA)(idx / (len_x - 1) - 1));
                idx = idx % (2 * (len_x - 1));
                if (idx < (len_x - 1))
                {
                    return re + (x[idx] - x[0]);
                }
                else
                {
                    return re + (x[len_x - 1] -
                                 x[len_x - 1 - (idx - (len_x - 1))]);
                }
            }
        case MODE_CONSTANT:
            return cval;
        default:
            return -1.;
    }
}

"""


_convolved_batch_template = (
    _include
    + r"""

extern "C" {

__global__
void _apply_batch(DTYPE_DATA *x, long long len_x,
               DTYPE_FILTER *h_trans_flip,
               long long len_h,
               DTYPE_OUT *out,
               long long out_axis_size,
               long long nbatch,
               long long _mode,
               DTYPE_DATA cval,
               long long offset,
               long long crop)
{
    __shared__ DTYPE_FILTER h_trans_flip_s[128];
    long long x_conv_idx;
    long long i;
    // TODO: set initial values for these constants outside the loop
    long long unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;

    for (i=0; i<len_h; i++)
    {
        h_trans_flip_s[i] = h_trans_flip[i];
    }

    if (batch_idx < nbatch)
    {
        long long padded_len;
        long long offset_x = batch_idx * len_x;
        long long offset_out = batch_idx * out_axis_size;

        long long y_idx = unraveled_idx - offset_out;
        long long h_idx = 0;
        long long x_idx = y_idx + offset;

        DTYPE_OUT val = 0.0;
        DTYPE_OUT xval;

        bool zpad = (mode == MODE_CONSTANT) && (abs(cval) == 0.0);

        if (crop)
        {
            padded_len = len_x + offset;
        }
        else
        {
            padded_len = len_x + len_h - 1 + offset;
        }

        if (x_idx < len_x)
        {
            x_conv_idx = x_idx - len_h + 1;
            if (x_conv_idx < 0){
                if (zpad)
                {
                    h_idx -= x_conv_idx;
                }
                else
                {
                    for (; x_conv_idx < 0; x_conv_idx++){
                        xval = _extend_left(
                            &x[offset_x], x_conv_idx, len_x, mode, cval);
                        val += xval * h_trans_flip_s[h_idx];
                        h_idx++;
                    }
                }
                x_conv_idx = 0;
            }
            for (; x_conv_idx < x_idx + 1; x_conv_idx++){
                val += x[offset_x + x_conv_idx] * h_trans_flip_s[h_idx];
                h_idx++;
            }
            atomicAdd(&out[unraveled_idx], val);
        }

        // Use a second simplified loop to flush out the last bits
        else if (x_idx < padded_len)
        {
            x_conv_idx = x_idx - len_h + 1;
            for (; x_conv_idx < x_idx + 1; x_conv_idx++)
            {
                if (x_conv_idx >= len_x)
                {
                    xval = _extend_right(
                        &x[offset_x], x_conv_idx, len_x, mode, cval);

                }
                else if (x_conv_idx < 0)
                {
                    xval = _extend_left(
                        &x[offset_x], x_conv_idx, len_x, mode, cval);
                }
                else
                {
                    xval = x[offset_x + x_conv_idx];
                }
                val += xval * h_trans_flip_s[h_idx];
                h_idx++;
            }
            atomicAdd(&out[unraveled_idx], val);
        }
    }
}
}
"""
)


_upfirdn_part1_shared_h = r"""

extern "C" {

__global__
void _apply_batch(DTYPE_DATA *x, long long len_x,
                  DTYPE_FILTER *h_trans_flip,
                  long long len_h,
                  DTYPE_OUT *out,
                  long long up,
                  long long down,
                  long long out_axis_size,
                  long long nbatch,
                  long long _mode,
                  DTYPE_DATA cval,
                  long long offset,
                  long long crop)
{
    __shared__ DTYPE_FILTER h_trans_flip_s[128];
    long long x_conv_idx;
    long long i;
    // TODO: set initial values for these constants outside the loop
    long long unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;

    // copy filter to shared memory
    for (i=0; i<len_h; i++)
    {
       h_trans_flip_s[i] = h_trans_flip[i];
    }
"""

_upfirdn_part1_nonshared_h = r"""

extern "C" {

__global__
void _apply_batch(DTYPE_DATA *x, long long len_x,
                  DTYPE_FILTER *h_trans_flip_s,
                  long long len_h,
                  DTYPE_OUT *out,
                  long long up,
                  long long down,
                  long long out_axis_size,
                  long long nbatch,
                  long long _mode,
                  DTYPE_DATA cval,
                  long long offset,
                  long long crop)
{
    long long x_conv_idx;
    long long i;
    // TODO: set initial values for these constants outside the loop
    long long unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;

"""

_upfirdn_template_part2 = r"""

    if (batch_idx < nbatch)
    {
        long long h_per_phase = len_h / up;
        long long padded_len;
        long long offset_x = batch_idx * len_x;
        long long offset_out = batch_idx * out_axis_size;

        long long y_idx = unraveled_idx - offset_out;
        long long t = ((y_idx + offset)*down) % up;
        long long h_idx = t * h_per_phase;
        long long x_idx = ((y_idx + offset)*down) / up;

        DTYPE_OUT val = 0.0;
        DTYPE_OUT xval;
        DTYPE_OUT hval = 0.0;

        bool zpad = (mode == MODE_CONSTANT) && (abs(cval) == 0.0);
        if (crop)
            padded_len = len_x + offset * down / up;
        else
            padded_len = len_x + len_h - 1 + offset * down / up;

        if (x_idx < len_x)
        {
            x_conv_idx = x_idx - h_per_phase + 1;
            if (x_conv_idx < 0){
                if (zpad)
                {
                    h_idx -= x_conv_idx;
                }
                else
                {
                    for (; x_conv_idx < 0; x_conv_idx++){
                        hval = h_trans_flip_s[h_idx];
                        if (hval != DTYPE_FILTER(0))
                        {
                            xval = _extend_left(
                                &x[offset_x], x_conv_idx, len_x, mode, cval);
                            val += xval * hval;
                        }
                        h_idx++;
                    }
                }
                x_conv_idx = 0;
            }
            for (; x_conv_idx < x_idx + 1; x_conv_idx++){
                hval = h_trans_flip_s[h_idx];
                if (hval != DTYPE_FILTER(0))
                {
                    val += x[offset_x + x_conv_idx] * hval;
                }
                h_idx++;
            }
            atomicAdd(&out[unraveled_idx], val);
        }

        // Use a second simplified loop to flush out the last bits
        else if (x_idx < padded_len)
        {
            x_conv_idx = x_idx - h_per_phase + 1;
            for (; x_conv_idx < x_idx + 1; x_conv_idx++)
            {
                hval = h_trans_flip_s[h_idx];
                if (hval != DTYPE_FILTER(0))
                {
                    if (x_conv_idx >= len_x)
                    {
                        xval = _extend_right(
                            &x[offset_x], x_conv_idx, len_x, mode, cval);

                    }
                    else if (x_conv_idx < 0)
                    {
                        xval = _extend_left(
                            &x[offset_x], x_conv_idx, len_x, mode, cval);
                    }
                    else
                    {
                        xval = x[offset_x + x_conv_idx];
                    }
                    val += xval * hval;
                }
                h_idx++;
            }
            atomicAdd(&out[unraveled_idx], val);
        }
    }
}
}
"""

# version where the filter, h, is copied into local shared memory
_upfirdn_batch_template_shared_h = (
    _include + _upfirdn_part1_shared_h + _upfirdn_template_part2
)

# version where the filter, h, is not copied into local shared memory
_upfirdn_batch_template_nonshared_h = (
    _include + _upfirdn_part1_nonshared_h + _upfirdn_template_part2
)


# dictionary: CUDA C data types corresponding to numpy dtype.char values
c_dtypes = {
    "f": "float",
    "d": "double",
    "F": "complex<float>",
    "D": "complex<double>",
}


@memoize()
def _nearest_supported_float_dtype(dtype):
    if dtype.char in ["f", "d", "F", "D"]:
        return dtype, c_dtypes.get(dtype.char)

    # determine nearest single or double precision floating point type
    dtype = np.result_type(dtype, np.float32)
    if dtype.char == "g":
        dtype = np.dtype(np.float64)
    elif dtype.char == "G":
        dtype = np.dtype(np.complex128)
    return dtype, c_dtypes.get(dtype.char)


@memoize(for_each_device=True)
def _get_upfirdn_kernel_inner(
    up, down, c_dtype_data, c_dtype_filter, c_dtype_out, h_size
):
    from cupy.cuda import nvrtc

    func_name = "_apply_batch"

    # crude template-like functionality via string replacement
    if up == down == 1:
        code = _convolved_batch_template.replace("DTYPE_DATA", c_dtype_data)
    else:
        if h_size <= 128:
            code = _upfirdn_batch_template_shared_h.replace(
                "DTYPE_DATA", c_dtype_data
            )
        else:
            code = _upfirdn_batch_template_nonshared_h.replace(
                "DTYPE_DATA", c_dtype_data
            )

    code = code.replace("DTYPE_FILTER", c_dtype_filter)
    code = code.replace("DTYPE_OUT", c_dtype_out)

    if nvrtc.getVersion() < (9, 2):
        # __shared__ complex<T> doesn't work on older CUDA compilers

        """ From the CUDA 9.2 release notes:
        The CUDA compiler previously incorrectly determined that the
        constructor for a __shared__ multidimensional array variable was
        non-empty in some scenarios, and generated a spurious diagnostic.
        The bug has now been fixed.
        """
        code = code.replace("__shared__ complex", "complex")

    kern = cupy.RawKernel(code, func_name)
    return kern


def get_upfirdn_kernel(h, data, up, down):
    """Compile an upfirdn kernel based on dtype.

    Also converts h, data to the nearest supported floating point type.
    """
    dtype_data, c_dtype_data = _nearest_supported_float_dtype(data.dtype)
    if data.dtype != dtype_data:
        data = data.astype(dtype_data)

    # convert h to the same precision as data if there is a mismatch
    if data.real.dtype != h.real.dtype:
        if np.iscomplexobj(h):
            h_dtype = np.result_type(data.real.dtype, np.complex64)
        else:
            h_dtype = np.result_type(data.real.dtype, np.float32)
        h = h.astype(h_dtype)

    dtype_filter, c_dtype_filter = _nearest_supported_float_dtype(h.dtype)
    if "complex" in c_dtype_filter:
        # output is complex if filter is complex
        c_dtype_out = c_dtype_filter
        dtype_out = dtype_filter
    else:
        # for real filter, output dtype matches the data dtype
        c_dtype_out = c_dtype_data
        dtype_out = dtype_data
    if h.dtype != dtype_filter:
        h = h.astype(dtype_filter)

    # memoized GPU kernels
    kern = _get_upfirdn_kernel_inner(
        up, down, c_dtype_data, c_dtype_filter, c_dtype_out, h.size
    )

    return h, data, dtype_out, kern


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.

    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].

    Then the internal buffer will look like this::

       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)

    Notes
    -----
    This is a copy of the function from scipy.signal._upfirdn.py with cupy
    support.
    """
    if up == 1:
        return h[::-1].copy()  # copy to avoid negative strides
    else:
        lh = len(h)
        h_padlen = lh + (-lh % up)
        if h_padlen == 0:
            h_full = h
        else:
            h_full = cupy.zeros(h_padlen, h.dtype)
            h_full[:lh] = h
        h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
        # TODO: numpy doesn't need this ascontiguousarray here,
        #       but cupy does to avoid negative strides!
        h_full = cupy.ascontiguousarray(h_full)
    return h_full


def _convolve1d(
    h,
    x,
    axis=-1,
    contiguous_output=False,
    block_size=32,
    out=None,
    mode="zero",
    cval=0,
    offset=0,
    crop=False,
):
    """

    out : use a preallocated output array.
        Unless filtering is being performed along the last axis, a copy may
        have to be made. For this reason, one should not rely on operation
        upon the ``out`` array to be in-place.
    """

    # compile or retrieve cached kernel for the given dtypes
    h, x, dtype_out, kern = get_upfirdn_kernel(h, x, up=1, down=1)

    ndim = x.ndim

    mode_enum = _get_mode_enum(mode)
    cval = dtype_out.type(cval)

    # flip the filter
    h_flip = h[::-1].copy()  # copy to avoid negative strides

    len_h = len(h_flip)
    if axis < -ndim or axis > ndim - 1:
        raise ValueError("axis out of range")
    axis = axis % x.ndim
    if axis != ndim - 1:
        x = x.swapaxes(axis, -1)
    out_shape = [s for s in x.shape]
    if not crop:
        out_len = x.shape[-1] + len_h - 1
    else:
        out_len = x.shape[-1]
    out_shape[-1] = out_len

    x = cupy.ascontiguousarray(x)
    x = x.reshape((-1, x.shape[-1]), order="C")
    nbatch = x.shape[0]

    if out is None:
        y = cupy.zeros((nbatch, out_len), dtype=dtype_out)
    else:
        # output into preallocated array
        if axis != ndim - 1:
            out = out.swapaxes(-1, axis)
        if out.size != nbatch * out_len or out.shape[-1] != out_len:
            raise ValueError("out array has the wrong size")
        elif out.dtype != dtype_out:
            raise ValueError(
                "Expected an out array with dtype: {}".format(dtype_out)
            )
        if not out.flags.c_contiguous:
            out = cupy.ascontiguousarray(out)
        out[:] = 0.0
        y = out

    grid_size_x = ceil(y.size / block_size)
    if grid_size_x > cuda_MaxGridDimX:
        raise ValueError(
            "Grid size > MaxGridDimX for the GPU. Try increasing block_size."
        )
    if block_size > cuda_MaxBlockDimX:
        raise ValueError("block_size exceeds MaxBlockDimX for the GPU")

    kern(
        (grid_size_x,),
        (block_size,),
        (
            x,
            x.shape[-1],
            h_flip,
            len_h,
            y,
            out_len,
            nbatch,
            mode_enum,
            cval,
            offset,
            crop,
        ),
    )
    y = y.reshape(out_shape, order="C")

    if axis != ndim - 1:
        y = y.swapaxes(axis, -1)
    if contiguous_output:
        y = cupy.ascontiguousarray(y)
    return y


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    contiguous_output=False,
    block_size=32,
    prepadded=False,
    out=None,
    mode="zero",
    cval=0,
    crop=False,
    take=None,
    offset=0,
):
    """

    out : use a preallocated output array.
        Unless filtering is being performed along the last axis, a copy may
        have to be made. For this reason, one should not rely on operation
        upon the ``out`` array to be in-place.
    """
    if not isinstance(h, cupy.ndarray):
        h = cupy.asarray(h)
    if not isinstance(x, cupy.ndarray):
        x = cupy.asarray(x)

    # compile or retrieve cached kernel for the given dtypes
    h, x, dtype_out, kern = get_upfirdn_kernel(h, x, up=up, down=down)

    ndim = x.ndim

    mode_enum = _get_mode_enum(mode)
    cval = dtype_out.type(cval)

    # flip the filter
    if prepadded:
        h_flip = h
    else:
        h_flip = _pad_h(h, up=up)

    len_h = len(h_flip)
    # if len_h > 128:
    #     raise ValueError(
    #         "CUDA implementation currently assumes filter length is <= 128."
    #     )

    if axis < -ndim or axis > ndim - 1:
        raise ValueError("axis out of range")
    axis = axis % x.ndim
    if axis != ndim - 1:
        x = x.swapaxes(axis, -1)
    out_shape = [s for s in x.shape]
    if crop:
        out_len = int(np.ceil(x.shape[-1] * up / down))
    else:
        out_len = _output_len(h_flip.size, x.shape[-1], up, down)
    out_shape[-1] = out_len

    x = cupy.ascontiguousarray(x)
    x = x.reshape((-1, x.shape[-1]), order="C")
    nbatch = x.shape[0]

    if out is None:
        y = cupy.zeros((nbatch, out_len), dtype=dtype_out)
    else:
        # output into preallocated array
        if axis != ndim - 1:
            out = out.swapaxes(-1, axis)
        if out.size != nbatch * out_len or out.shape[-1] != out_len:
            raise ValueError("out array has the wrong size")
        elif out.dtype != dtype_out:
            raise ValueError(
                "Expected an out array with dtype: {}".format(dtype_out)
            )
        if not out.flags.c_contiguous:
            out = cupy.ascontiguousarray(out)
        out[:] = 0.0
        y = out

    grid_size_x = ceil(y.size / block_size)
    if grid_size_x > cuda_MaxGridDimX:
        raise ValueError(
            "Grid size > MaxGridDimX for the GPU. Try increasing block_size."
        )
    if block_size > cuda_MaxBlockDimX:
        raise ValueError("block_size exceeds MaxBlockDimX for the GPU")
    if up == down == 1:
        kern(
            (grid_size_x,),
            (block_size,),
            (
                x,
                x.shape[-1],
                h_flip,
                len_h,
                y,
                out_len,
                nbatch,
                mode_enum,
                cval,
                offset,
                int(crop),
            ),
        )
    else:
        kern(
            (grid_size_x,),
            (block_size,),
            (
                x,
                x.shape[-1],
                h_flip,
                len_h,
                y,
                up,
                down,
                out_len,
                nbatch,
                mode_enum,
                cval,
                offset,
                int(crop),
            ),
        )
    y = y.reshape(out_shape, order="C")

    if take is not None:
        # TODO: move into the kernel
        y_sl = [slice(None)] * y.ndim
        if take is None:
            y_sl[-1] = slice(None)
        else:
            if isinstance(take, slice):
                y_sl[-1] = take
            else:
                y_sl[-1] = slice(take)
        y = y[tuple(y_sl)]

    if axis != ndim - 1:
        y = y.swapaxes(axis, -1)
    if contiguous_output:
        y = cupy.ascontiguousarray(y)
    return y
