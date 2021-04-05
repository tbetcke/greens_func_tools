"""A library for the evaluation of Greens functions."""
from .greens_func_tools import ffi, lib

def test():
    """Test the library."""
    import pytest
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))

    pytest.main([current_dir])
    