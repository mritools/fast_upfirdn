# TODO: finalize behavior of offset, crop, etc.
#       currently passing tests as-is, but the implementation could be cleaner

import warnings
from math import ceil

import numpy as np
import cupy
from cupy.util import memoize


try:
    # Device Attributes require CuPy > 6.0.b3
    import cupy
    d = cupy.cuda.device.Device(0)
    cuda_MaxBlockDimX = d.attributes['MaxBlockDimX']
    cuda_MaxGridDimX = d.attributes['MaxGridDimX']
except AttributeError:
    # guess
    cuda_MaxBlockDimX = 1024
    cuda_MaxGridDimX = 2147483647


try:
    from scipy.signal._upfirdn import _output_len
except ImportError:
    def _output_len(len_h, in_len, up, down):
        """The output length that results from a given input"""
        in_len_copy = in_len + (len_h + (-len_h % up)) // up - 1
        nt = in_len_copy * up
        need = nt // down
        if nt % down > 0:
            need += 1
        return need


include = r"""
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
    MODE_ZEROPAD = 0,
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
        return x[(len_x + idx) % len_x];
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
        return cval;
    case MODE_ZEROPAD:
        return 0.;
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
                    return x[len_x - (idx - len_x)];
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
            return x[(idx - len_x) % len_x];
        }
        case MODE_SMOOTH:
            return x[len_x - 1] +
                   (DTYPE_DATA)(idx - len_x) *
                   (x[len_x - 1] - x[len_x - 2]);
        case MODE_CONSTANT_EDGE:
            return cval;
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
                    return -x[idx];
                }
                else
                {
                    return x[len_x - (idx - len_x)];
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
        case MODE_ZEROPAD:
            return 0.;
        default:
            return -1.;
    }
}

"""


# TESTED vs. CPU
_apply_batch_up1_template = include + r"""

extern "C"
{

__global__
void _apply_batch_up1(DTYPE_DATA *x, long long len_x,
                      DTYPE_FILTER *h_trans_flip,
                      long long len_h,
                      DTYPE_OUT *out,
                      long long down,
                      long long out_axis_size,
                      long long nbatch,
                      long long _mode,
                      DTYPE_DATA cval,
                      long long offset,
                      long long crop)
{
    __shared__ DTYPE_FILTER h_trans_flip_s[128];
    long long i;
    long long unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long batch_idx = unraveled_idx / out_axis_size;
    MODE mode = (MODE)_mode;
    DTYPE_OUT val = 0.0;

    for (i=0; i<len_h; i++)
    {
        h_trans_flip_s[i] = h_trans_flip[i];
    }

    if (batch_idx < nbatch)
    {
        long long padded_len;
        long long h_idx = 0;
        long long x_conv_idx = 0;
        long long offset_x = batch_idx * len_x;
        long long offset_out = batch_idx * out_axis_size;
        long long y_idx = unraveled_idx - offset_out;
        long long x_idx = down * y_idx;
        DTYPE_OUT xval;

        bool zpad = (mode == MODE_ZEROPAD);
        if (crop)
            padded_len = len_x;
        else
            padded_len = len_x + len_h - 1;

        if (x_idx < offset)
            return;

        if (x_idx < len_x)
        {
            h_idx = 0;
            x_conv_idx = x_idx - len_h + 1;
            if (x_conv_idx < 0){
                if (zpad)
                {
                    h_idx -= x_conv_idx;
                }
                else
                {
                    for (; x_conv_idx < 0; x_conv_idx++){
                        xval = _extend_left(&x[offset_x], x_conv_idx, len_x, mode, cval);
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
            h_idx = 0;
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


# TESTED vs. CPU
_apply_batch_down1_template = include + """

extern "C" {

__global__
void _apply_batch_down1(DTYPE_DATA *x, long long len_x,
                        DTYPE_FILTER *h_trans_flip,
                        long long len_h,
                        DTYPE_OUT *out,
                        long long up,
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
        long long h_per_phase = len_h / up;
        long long padded_len;
        long long offset_x = batch_idx * len_x;
        long long offset_out = batch_idx * out_axis_size;
        long long y_idx = unraveled_idx - offset_out;
        long long x_idx = (y_idx + offset) / up;
        long long t = (y_idx + offset) % up;
        long long h_idx = t * h_per_phase;
        DTYPE_OUT val = 0.0;
        DTYPE_OUT xval;

        bool zpad = ((mode == MODE_ZEROPAD));
        if (crop)
            padded_len = len_x;
        else
            padded_len = len_x + len_h - 1;

        //if (x_idx < (offset / up))
        //    return;

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
            x_conv_idx = x_idx - h_per_phase + 1;
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


c_dtypes = {'f': 'float',
            'd': 'double',
            'F': 'complex<float>',
            'D': 'complex<double>'}


def _fixup_dtype(dtype):
    dtype_char = dtype.char
    if dtype_char in ['f', 'd', 'F', 'D']:
        return dtype, c_dtypes.get(dtype_char)

    dtype = np.result_type(dtype, np.float32)
    dtype_char = dtype.char
    if dtype_char == 'g':
        msg = "float128 not supported. using float64 instead"
        warnings.warn(msg)
        dtype = np.float64
    elif dtype_char == 'G':
        msg = "complex256 not supported. using complex128 instead"
        warnings.warn(msg)
        dtype = np.complex128
    c_dtype = c_dtypes.get(dtype_char, None)
    if c_dtype is None:
        raise ValueError("unsupported dtype: {}".format(dtype))
    return dtype, c_dtype


def _get_mode_enum(mode):
    mode = mode.lower()
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'constant':
        return 2
    elif mode == 'smooth':
        return 3
    elif mode == 'periodic':
        return 4
    elif mode == 'reflect':
        return 5
    elif mode == 'antisymmetric':
        return 6
    elif mode == 'antireflect':
        return 7
    else:
        raise ValueError("Unknown mode: {}".format(mode))


@memoize(for_each_device=True)
def _get_upfirdn_kernel_inner(up, down, case, c_dtype_data, c_dtype_filter,
                              c_dtype_out):
    if up == 1:
        code = _apply_batch_up1_template.replace(
            'DTYPE_DATA', c_dtype_data)
        func_name = '_apply_batch_up1'
    elif down == 1:
        code = _apply_batch_down1_template.replace(
            'DTYPE_DATA', c_dtype_data)
        func_name = '_apply_batch_down1'
    else:
        raise ValueError(
            "CUDA kernels only implemented for cases with down=1 or up=1.")
    code = code.replace(
        'DTYPE_FILTER', c_dtype_filter)
    code = code.replace(
        'DTYPE_OUT', c_dtype_out)

    if cupy.cuda.nvrtc.getVersion() < (9, 2):
        # __shared__ complex<T> doesn't work on older CUDA compilers

        """ From the CUDA 9.2 release notes:
        The CUDA compiler previously incorrectly determined that the
        constructor for a __shared__ multidimensional array variable was
        non-empty in some scenarios, and generated a spurious diagnostic.
        The bug has now been fixed.
        """
        code = code.replace('__shared__ complex', 'complex')

    kern = cupy.RawKernel(code, func_name)
    return kern


def get_upfirdn_kernel(h, data, up, down):
    dtype_data, c_dtype_data = _fixup_dtype(data.dtype)
    if data.dtype != dtype_data:
        data = data.astype(dtype_data)

    # convert h to the same precision as data if there is a mismatch
    if data.real.dtype != h.real.dtype:
        if np.iscomplexobj(h):
            h_dtype = np.result_type(data.real.dtype, np.complex64)
        else:
            h_dtype = np.result_type(data.real.dtype, np.float32)
        h = h.astype(h_dtype)

    dtype_filter, c_dtype_filter = _fixup_dtype(h.dtype)
    if 'complex' in c_dtype_filter:
        # output is complex if filter is complex
        c_dtype_out = c_dtype_filter
        dtype_out = dtype_filter
    else:
        # for real filter, output dtype matches the data dtype
        c_dtype_out = c_dtype_data
        dtype_out = dtype_data
    if h.dtype != dtype_filter:
        h = h.astype(dtype_filter)
    if up == 1:
        case = 0
    elif down == 1:
        case = 1
    else:
        raise ValueError(
            "CUDA kernels only implemented for cases with down=1 or up=1.")
    kern = _get_upfirdn_kernel_inner(up, down, case, c_dtype_data,
                                     c_dtype_filter, c_dtype_out)

    return h, data, dtype_out, kern


def upfirdn(h, x, up=1, down=1, axis=-1, contiguous_output=False,
            block_size=32, prepadded=False, out=None, mode='zero',
            cval=0, offset=0, crop=0, take=None, h_size_orig=None):
    # compile or retrieve cached kernel for the given dtypes
    h, x, dtype_out, kern = get_upfirdn_kernel(h, x, up=up, down=down)

    ndim = x.ndim

    mode_enum = _get_mode_enum(mode)
    cval = dtype_out.type(cval)
    crop = int(crop)  # TODO: remove hardcode

    offset_here = True
    if offset_here:
        offset = 0  # TODO: remove hardcode

    # flip the filter
    if prepadded:
        h_flip = h
    else:
        from pyframelets._num import _pad_h
        h_flip = _pad_h(h, up=up, xp=cupy)

    len_h = len(h_flip)
    if len_h > 128:
        raise ValueError(
            "CUDA implementation currently assumes filter length is <= 128.")

    if axis < -ndim or axis > ndim - 1:
        raise ValueError("axis out of range")
    axis = axis % x.ndim
    if axis != ndim - 1:
        x = x.swapaxes(axis, -1)
    out_shape = [s for s in x.shape]
    out_len = _output_len(h_flip.size, x.shape[-1], up, down) - offset
    out_shape[-1] = out_len

    x = cupy.ascontiguousarray(x)
    x = x.reshape((-1, x.shape[-1]), order='C')
    nbatch = x.shape[0]

    if out is None:
        y = cupy.zeros((nbatch, out_len), dtype=dtype_out)
    else:
        # output into preallocated array
        if out.size != nbatch * out_len:
            raise ValueError("out array has the wrong size")
        elif not out.flags.c_contiguous:
            raise ValueError("out array must be C contiguous")
        y = out

    grid_size_x = ceil(y.size / block_size)
    if grid_size_x > cuda_MaxGridDimX:
        raise ValueError(
            "Grid size > MaxGridDimX for the GPU. Try increasing block_size.")
    if block_size > cuda_MaxBlockDimX:
        raise ValueError("block_size exceeds MaxBlockDimX for the GPU")
    if up == 1:
        kern((grid_size_x, ),
             (block_size, ),
             (x, x.shape[-1], h_flip, len_h, y, down, out_len, nbatch,
              mode_enum, cval, offset, crop))
    elif down == 1:
        kern((grid_size_x, ),
             (block_size, ),
             (x, x.shape[-1], h_flip, len_h, y, up, out_len, nbatch,
              mode_enum, cval, offset, crop))
    y = y.reshape(out_shape, order='C')

    if crop:
        if offset_here:
            # TODO: move into the kernel
            if h_size_orig is None:
                offset = len_h - 1
            else:
                offset = h_size_orig - 1
            # print("offset = {}, len_h = {}, up={}, len(h)={}".format(offset, len_h, up, len(h)))
            y_sl = [slice(None), ] * y.ndim
            if take is None:
                y_sl[-1] = slice(offset, None)
            else:
                y_sl[-1] = slice(offset, offset + take)
            y = y[tuple(y_sl)]
        else:
            y_sl = [slice(None), ] * y.ndim
            if take is not None:
                y_sl[-1] = slice(take)
                y = y[tuple(y_sl)]

    if axis != ndim - 1:
        y = y.swapaxes(axis, -1)
    if contiguous_output:
        y = cupy.ascontiguousarray(y)
    return y
