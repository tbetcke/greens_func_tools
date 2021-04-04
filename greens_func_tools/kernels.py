import numpy as np
from greens_func_tools import ffi, lib

def _as_double_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast('double*', arr.ctypes.data)

def _to_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)


def print_value():
    """Test print a value."""

    import numpy as np

    arr = np.array([[1.0, 2.0]], dtype=np.float64)

    rows, cols = arr.shape

    lib.rust_function(_as_double_ptr(arr), _to_usize(rows), _to_usize(cols))
