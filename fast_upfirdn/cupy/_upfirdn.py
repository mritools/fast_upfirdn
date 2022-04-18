"""GPU Implementation of upfirdn.

A separate implementation "convolve1d" is also provided as a somewhat faster
alternative in the case up=down=1 (i.e. standard convolution).
"""
from math import ceil

import numpy as np

import cupy

try:
    from cupy.util import memoize
except ImportError:
    from cupy import memoize
from fast_upfirdn.cpu._upfirdn_apply import mode_enum as _get_mode_enum
from fast_upfirdn._util import profile


try:
    # Device Attributes require CuPy > 6.0.b3
    d = cupy.cuda.Device()
    cuda_MaxBlockDimX = d.attributes["MaxBlockDimX"]
    cuda_MaxGridDimX = d.attributes["MaxGridDimX"]
except (cupy.cuda.runtime.CUDARuntimeError, AttributeError):
    # guess
    cuda_MaxBlockDimX = 1024
    cuda_MaxGridDimX = 2147483647

__all__ = ["convolve1d", "upfirdn"]


def _output_len(len_h, in_len, up, down):
    """The output length that results from a given input.

    scipy.signal._upfirdn._output_len
    """
    return (((in_len - 1) * up + len_h) - 1) // down + 1


_include = r"""
#include "cupy/complex.cuh"
#include "cupy/carray.cuh"

typedef {data_type} data_t;
typedef {index_type} idx_t;
typedef {filter_type} filter_t;
typedef {out_type} out_t;
"""

_UPFIRDN_COMMON = r"""
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
    MODE_ANTIREFLECT = 7,
    MODE_LINE = 8
};


__device__
data_t _extend_left(data_t *x, idx_t idx, idx_t len_x, MODE mode, data_t cval)
{
    data_t le = 0.;
    data_t lin_slope = 0.;

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
        return x[0] + (data_t)idx * (x[1] - x[0]);
    case MODE_LINE:
        lin_slope = (x[len_x - 1] - x[0]) / (data_t)(len_x - 1);
        return x[0] + (data_t)idx * lin_slope;
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
                 ((data_t)((-(idx) - 1) / (len_x - 1)));
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
data_t _extend_right(data_t *x, idx_t idx, idx_t len_x, MODE mode, data_t cval)
{
    // note: idx will be >= len_x
    data_t re = 0.;
    data_t lin_slope = 0.;
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
                   (data_t)(idx - len_x + 1) *
                   (x[len_x - 1] - x[len_x - 2]);
        case MODE_LINE:
            lin_slope = (x[len_x - 1] - x[0]) / (data_t)(len_x - 1);
            return x[len_x - 1] + (data_t)(idx - len_x + 1) * lin_slope;
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
                     ((data_t)(idx / (len_x - 1) - 1));
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


_convolved_batch = r"""

extern "C" {

__global__
void _apply_batch(data_t *x, idx_t len_x,
                  filter_t *h_trans_flip_s,
                  idx_t len_h,
                  out_t *out,
                  idx_t out_axis_size,
                  idx_t nbatch,
                  int _mode,
                  data_t cval,
                  int offset,
                  int crop)
{
    idx_t x_conv_idx;
    idx_t i;
    // TODO: set initial values for these constants outside the loop
    idx_t unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;

    if (batch_idx < nbatch)
    {
        idx_t padded_len;
        idx_t offset_x = batch_idx * len_x;
        idx_t offset_out = batch_idx * out_axis_size;

        idx_t h_idx = 0;
        idx_t x_idx = unraveled_idx - offset_out + offset;

        out_t val = 0.0;
        out_t xval;

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
            if (x_conv_idx < 0)
            {
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

_upfirdn_h = r"""

extern "C" {

__global__
void _apply_batch(data_t *x, idx_t len_x,
                  filter_t *h_trans_flip_s,
                  idx_t len_h,
                  out_t *out,
                  int up,
                  int down,
                  idx_t out_axis_size,
                  idx_t nbatch,
                  int _mode,
                  data_t cval,
                  int offset,
                  int crop)
{
    idx_t x_conv_idx;
    idx_t i;
    // TODO: set initial values for these constants outside the loop
    idx_t unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;

    if (batch_idx < nbatch)
    {
        idx_t h_per_phase = len_h / up;
        idx_t padded_len;
        idx_t offset_x = batch_idx * len_x;
        idx_t offset_out = batch_idx * out_axis_size;

        idx_t tmp_idx = unraveled_idx - offset_out + offset;
        idx_t t = (tmp_idx * down) % up;
        idx_t x_idx = (tmp_idx * down) / up;
        idx_t h_idx = t * h_per_phase;

        out_t val = 0.0;
        out_t xval;
        out_t hval = 0.0;

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
                        if (hval != filter_t(0))
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
                if (hval != filter_t(0))
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
                if (hval != filter_t(0))
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

# version where the filter, h, is not copied into local shared memory
_upfirdn_batch_nonshared_h = _upfirdn_h


# dictionary: CUDA C data types corresponding to numpy dtype.char values
c_dtypes = {
    "f": "float",
    "d": "double",
    "F": "complex<float>",
    "D": "complex<double>",
}


@memoize()
def _nearest_supported_float_dtype(dtype, dtype2=None):
    if dtype.char in ["f", "d", "F", "D"] and (
        dtype2 is None or dtype2 == dtype
    ):
        return dtype, c_dtypes.get(dtype.char)

    # determine nearest single or double precision floating point type
    dtype = np.promote_types(dtype, np.float32)
    if dtype2 is not None:
        dtype = np.promote_types(dtype, dtype2)
    if dtype.char == "g":
        dtype = np.dtype(np.float64)
    elif dtype.char == "G":
        dtype = np.dtype(np.complex128)
    return dtype, c_dtypes.get(dtype.char)


@memoize(for_each_device=True)
def _get_upfirdn_kernel_inner(
    up,
    down,
    c_data_type,
    c_filter_type,
    c_out_type,
    h_size,
    c_index_type,
):
    func_name = "_apply_batch"
    code = _include.format(
        data_type=c_data_type,
        filter_type=c_filter_type,
        out_type=c_out_type,
        index_type=c_index_type,
    )
    if up == down == 1:
        code += _UPFIRDN_COMMON + _convolved_batch
    else:
        code += _UPFIRDN_COMMON + _upfirdn_batch_nonshared_h
    return cupy.RawKernel(code, func_name)


@memoize()
def _determine_dtypes(data_dtype, data_real_dtype, h_dtype, h_real_dtype):
    # note need to include h_real_dtype here so data's dtype will be promoted
    # if h has a higher precision dtype.
    data_type, c_data_type = _nearest_supported_float_dtype(
        data_dtype, h_real_dtype
    )
    if data_dtype == h_dtype:
        out_type = filter_type = data_type
        c_out_type = c_filter_type = c_data_type
    else:
        # convert h to the same precision as data if there is a mismatch
        cplx_h = h_dtype.kind == "c"
        if data_real_dtype != h_real_dtype:
            if cplx_h:
                h_dtype = np.promote_types(data_type, np.complex64)
            else:
                h_dtype = np.promote_types(data_type, np.float32)

        filter_type, c_filter_type = _nearest_supported_float_dtype(h_dtype)
        if cplx_h:
            # output is complex if filter is complex
            c_out_type = c_filter_type
            out_type = filter_type
        else:
            # for real filter, output dtype matches the data dtype
            c_out_type = c_data_type
            out_type = data_type
    return (
        data_type,
        c_data_type,
        filter_type,
        c_filter_type,
        out_type,
        c_out_type,
    )


@profile
def get_upfirdn_kernel(h, data, up, down, c_index_type="int"):
    """Compile an upfirdn kernel based on dtype.

    Also converts h, data to the nearest supported floating point type.
    """
    dt_data, c_dt_data, dt_h, c_dt_h, dt_out, c_dt_out = _determine_dtypes(
        data.dtype, data.real.dtype, h.dtype, h.real.dtype
    )

    if data.dtype != dt_data:
        data = data.astype(dt_data)

    if h.dtype != dt_h:
        h = h.astype(dt_h)

    # memoized GPU kernels
    kern = _get_upfirdn_kernel_inner(
        up, down, c_dt_data, c_dt_h, c_dt_out, h.size, c_index_type
    )

    return h, data, dt_out, kern


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


def convolve1d(
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
    size_max = max(x.size, h.size)
    c_index_type = "size_t" if size_max > 1 << 31 else "int"
    h, x, out_type, kern = get_upfirdn_kernel(
        h, x, up=1, down=1, c_index_type=c_index_type
    )

    ndim = x.ndim

    mode_enum = _get_mode_enum(mode)
    cval = out_type.type(cval)

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

    if out is not None and out is x:
        # avoid clobbering existing values in x
        x = x.copy()
    else:
        x = cupy.ascontiguousarray(x)
    x = x.reshape((-1, x.shape[-1]), order="C")
    nbatch = x.shape[0]

    inplace_output = False
    if out is None:
        y = cupy.zeros((nbatch, out_len), dtype=out_type)
        out = y
    else:
        _out_orig = out
        # output into preallocated array
        if axis != ndim - 1:
            out = out.swapaxes(-1, axis)
        if out.size != nbatch * out_len or out.shape[-1] != out_len:
            raise ValueError("out array has the wrong size")
        elif out.dtype != out_type:
            raise ValueError(
                "Expected an out array with dtype: {}".format(out_type)
            )
        if not out.flags.c_contiguous:
            out = cupy.ascontiguousarray(out)
        out[:] = 0.0
        y = out
        inplace_output = True

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
    if inplace_output:
        if _out_orig is not y:
            if _out_orig.shape != y.shape:
                raise ValueError(
                    "output array does not have the expected shape"
                )
            _out_orig[...] = y[...]
        return _out_orig
    return y


@profile
def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    mode="zero",
    cval=0,
    *,
    contiguous_output=False,
    block_size=32,
    prepadded=False,
    out=None,
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

    # dev = cupy.cuda.Device()

    if down < 1 or up < 1:
        raise ValueError("Both up and down must be >= 1")
    if h.ndim != 1 or h.size == 0:
        raise ValueError("h must be 1-D with non-zero length")

    # compile or retrieve cached kernel for the given dtypes
    size_max = max(x.size, h.size)
    c_index_type = "size_t" if size_max > 1 << 31 else "int"
    h, x, out_type, kern = get_upfirdn_kernel(
        h, x, up=up, down=down, c_index_type=c_index_type
    )

    ndim = x.ndim

    mode_enum = _get_mode_enum(mode)
    cval = out_type.type(cval)

    # flip the filter
    if prepadded:
        h_flip = h
    else:
        h_flip = _pad_h(h, up=up)

    len_h = len(h_flip)

    if axis < -ndim or axis > ndim - 1:
        raise ValueError("axis out of range")
    axis = axis % x.ndim
    if axis != ndim - 1:
        x = x.swapaxes(axis, -1)
    out_shape = [s for s in x.shape]
    if crop:
        out_len = int(ceil(x.shape[-1] * up / down))
    else:
        out_len = _output_len(len(h), x.shape[-1], up, down)
    out_shape[-1] = out_len

    if out is not None and out is x:
        # avoid clobbering existing values in x
        x = x.copy()
    else:
        x = cupy.ascontiguousarray(x)
    x = x.reshape((-1, x.shape[-1]), order="C")
    nbatch = x.shape[0]

    inplace_output = False
    if out is None:
        y = cupy.zeros((nbatch, out_len), dtype=out_type)
        out = y
    else:
        _out_orig = out
        # output into preallocated array
        if axis != ndim - 1:
            out = out.swapaxes(-1, axis)
        if out.size != nbatch * out_len or out.shape[-1] != out_len:
            raise ValueError("out array has the wrong size")
        elif out.dtype != out_type:
            raise ValueError(
                "Expected an out array with dtype: {}".format(out_type)
            )
        if not out.flags.c_contiguous:
            out = cupy.ascontiguousarray(out)
        out[:] = 0.0
        y = out
        inplace_output = True

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
    if inplace_output:
        if _out_orig is not y:
            if _out_orig.shape != y.shape:
                raise ValueError(
                    "output array does not have the expected shape"
                )
            _out_orig[...] = y[...]
        return _out_orig
    return y
