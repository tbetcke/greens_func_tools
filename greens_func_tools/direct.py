import numpy as np
from greens_func_tools import ffi, lib


def _as_double_ptr(arr):
    """Turn to a double ptr."""
    return ffi.cast("double*", arr.ctypes.data)


def _as_float_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast("float*", arr.ctypes.data)


def _as_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)


def _align_data(arr, dtype=None):
    """Make sure that an array has the right properties."""

    if dtype is None:
        dtype = arr.dtype

    return np.require(arr, dtype=dtype, requirements=["C", "A"])


def assemble_laplace_kernel(targets, sources, dtype=np.float64, parallel=True):
    """Assemble the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    target = _align_data(targets, dtype=dtype)
    sources = _align_data(sources, dtype=dtype)

    result = np.empty((ntargets, nsources), dtype=dtype)

    if dtype == np.float32:
        lib.assemble_laplace_kernel_f32(
            _as_float_ptr(targets),
            _as_float_ptr(sources),
            _as_float_ptr(result),
            _as_usize(nsources),
            _as_usize(ntargets),
            parallel,
        )
    elif dtype == np.float64:
        lib.assemble_laplace_kernel_f64(
            _as_double_ptr(targets),
            _as_double_ptr(sources),
            _as_double_ptr(result),
            _as_usize(nsources),
            _as_usize(ntargets),
            parallel,
        )
    else:
        raise NotImplementedError

    return result

def evaluate_laplace_kernel(targets, sources, charges, dtype=np.float64, parallel=True):
    """Evaluate the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    if charges.ndim != 1 or charges.shape[0] != sources.shape[1]:
        raise ValueError(
            f"charges must be a 1-dim array of shape (nsources, ), current shape: {charges.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    target = _align_data(targets, dtype=dtype)
    sources = _align_data(sources, dtype=dtype)
    charges = _align_data(charges, dtype=dtype)

    result = np.empty(ntargets, dtype=dtype)

    if dtype == np.float32:
        lib.evaluate_laplace_kernel_f32(
            _as_float_ptr(targets),
            _as_float_ptr(sources),
            _as_float_ptr(charges),
            _as_float_ptr(result),
            _as_usize(nsources),
            _as_usize(ntargets),
            parallel,
        )
    elif dtype == np.float64:
        lib.evaluate_laplace_kernel_f64(
            _as_double_ptr(targets),
            _as_double_ptr(sources),
            _as_double_ptr(charges),
            _as_double_ptr(result),
            _as_usize(nsources),
            _as_usize(ntargets),
            parallel,
        )
    else:
        raise NotImplementedError

    return result

