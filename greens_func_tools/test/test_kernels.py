"""Unit tests for kernels."""
import numpy as np
import pytest


@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 1e-6)])
def test_laplace_kernel(dtype, rtol):
    """Test the Laplace kernel."""
    from greens_func_tools.kernels import laplace_kernel

    nsources = 10

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    target = 1.5 + rng.random(3, dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = target[:]  # Test what happens if source = target

    actual = laplace_kernel(target, sources, dtype=dtype)

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = 1.0 / (
        4 * np.pi * np.linalg.norm(sources - target.reshape(3, 1), axis=0)
    )

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0] = 0  # First source and target are identical.

    np.testing.assert_allclose(actual, expected, rtol=rtol)
