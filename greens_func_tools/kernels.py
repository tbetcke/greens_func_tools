import numpy as np
from greens_func_tools import ffi, lib

def _as_double_ptr(arr):
    """Turn to a double ptr."""
    return ffi.cast('double*', arr.ctypes.data)

def _as_float_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast('float*', arr.ctypes.data)


def _as_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)

def _align_data(arr, dtype=None):
    """Make sure that an array has the right properties."""
    
    if dtype is None:
        dtype = arr.dtype

    return np.require(arr, dtype=dtype, requirements=['C', 'A'])

def laplace_kernel(target, sources, dtype=np.float64):
    """Evaluate the Laplace kernel for one target and many sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(f"dtype must be one of [np.float64, np.float32], current value: {dtype}.")

    if target.ndim > 1 and target.shape[0] != 3:
        raise ValueError(f"target must be a 1-dim array of length 3, current shape: {target.shape}.")

    if sources.ndim != 2 and sources.shape[0] != 3:
        raise ValueError(f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}.")
        
    nsources = sources.shape[1]

    target = _align_data(target, dtype=dtype)
    sources = _align_data(sources, dtype=dtype)

    result = np.empty(nsources, dtype=dtype)

    if dtype == np.float32:
        lib.laplace_kernel_f32(_as_float_ptr(target), _as_float_ptr(sources), _as_float_ptr(result), _as_usize(nsources))
    elif dtype == np.float64:
        lib.laplace_kernel_f64(_as_double_ptr(target), _as_double_ptr(sources), _as_double_ptr(result), _as_usize(nsources))
    else:
        raise NotImplementedError

    return result


