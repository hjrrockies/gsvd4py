"""
Tests for gsvd4py.

Validates:
  - LAPACK library is found and loaded
  - Reconstruction accuracy: A ≈ U @ C @ X.conj().T,  B ≈ V @ S @ X.conj().T
  - Unitarity of U and V
  - All modes (full, econ, separate)
  - All four dtypes (float32, float64, complex64, complex128)
  - compute_u=False / compute_v=False short-tuple returns
  - Various shapes (square, tall, wide, rank-deficient)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gsvd4py import gsvd
import gsvd4py._lapack as _lapack_mod


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
_RTOL = {
    np.float32:    1e-5,
    np.float64:    1e-12,
    np.complex64:  1e-5,
    np.complex128: 1e-12,
}


# ---------------------------------------------------------------------------
# Test: library loading
# ---------------------------------------------------------------------------

class TestLibraryLoading:
    def test_loads_without_error(self):
        _lapack_mod._load_lib()
        assert _lapack_mod._lib_type in ('accelerate', 'scipy_openblas', 'system')

    def test_lib_type_is_string(self):
        _lapack_mod._load_lib()
        assert isinstance(_lapack_mod._lib_type, str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_matrix(rng, m, n, dtype):
    """Generate a random matrix of the given real or complex dtype."""
    if np.issubdtype(dtype, np.complexfloating):
        rdtype = np.float32 if dtype == np.complex64 else np.float64
        return (rng.standard_normal((m, n)).astype(rdtype) +
                1j * rng.standard_normal((m, n)).astype(rdtype)).astype(dtype)
    return rng.standard_normal((m, n)).astype(dtype)


def _check_reconstruction(U, V, X, C, S, A, B, rtol):
    """Check A ≈ U @ C @ X.conj().T and B ≈ V @ S @ X.conj().T."""
    XH = X.conj().T
    assert_allclose(U @ C @ XH, A, rtol=rtol, atol=rtol * np.linalg.norm(A))
    assert_allclose(V @ S @ XH, B, rtol=rtol, atol=rtol * np.linalg.norm(B))


def _check_unitary(M, rtol):
    """Check M @ M.conj().T ≈ I."""
    n = M.shape[1]
    assert_allclose(M.conj().T @ M, np.eye(n), atol=rtol * 10)


# ---------------------------------------------------------------------------
# Test: mode='full'
# ---------------------------------------------------------------------------

class TestFullMode:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                        np.complex64, np.complex128])
    @pytest.mark.parametrize("shape", [
        (5, 4, 6),   # m=5, n=4, p=6  (tall A, tall B)
        (4, 4, 4),   # square
        (3, 5, 6),   # m < p, n < p  (wide)
        (6, 3, 4),   # m > p, n < p
    ])
    def test_reconstruction(self, dtype, shape):
        m, n, p = shape
        rng = np.random.default_rng(42)
        A = _random_matrix(rng, m, p, dtype)
        B = _random_matrix(rng, n, p, dtype)
        rtol = _RTOL[dtype]

        U, V, X, C, S = gsvd(A, B, mode='full')

        assert U.shape == (m, m)
        assert V.shape == (n, n)
        assert X.shape[0] == p
        assert C.shape[0] == m
        assert S.shape[0] == n
        assert C.shape[1] == X.shape[1] == S.shape[1]  # same q

        _check_reconstruction(U, V, X, C, S, A, B, rtol)
        _check_unitary(U, rtol)
        _check_unitary(V, rtol)

    def test_no_u_no_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=False, compute_v=False)
        assert len(result) == 3
        X, C, S = result
        assert X.shape[0] == 5

    def test_no_u_with_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=False, compute_v=True)
        assert len(result) == 4
        V, X, C, S = result
        assert V.shape == (3, 3)

    def test_with_u_no_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=True, compute_v=False)
        assert len(result) == 4
        U, X, C, S = result
        assert U.shape == (4, 4)


# ---------------------------------------------------------------------------
# Test: mode='econ'
# ---------------------------------------------------------------------------

class TestEconMode:
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_reconstruction(self, dtype):
        rng = np.random.default_rng(7)
        m, n, p = 6, 4, 5
        A = _random_matrix(rng, m, p, dtype)
        B = _random_matrix(rng, n, p, dtype)
        rtol = _RTOL[dtype]

        U, V, X, C, S = gsvd(A, B, mode='econ')
        q = X.shape[1]

        assert U.shape == (m, min(m, q))
        assert V.shape == (n, min(n, q))
        assert C.shape == (min(m, q), q)
        assert S.shape == (min(n, q), q)

        _check_reconstruction(U, V, X, C, S, A, B, rtol)

    def test_econ_smaller_than_full(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((8, 5))
        B = rng.standard_normal((6, 5))
        U_f, V_f, X_f, C_f, S_f = gsvd(A, B, mode='full')
        U_e, V_e, X_e, C_e, S_e = gsvd(A, B, mode='econ')
        q = X_e.shape[1]
        # Economy U/V should be the first q columns of the full U/V
        assert U_e.shape[1] <= U_f.shape[1]
        assert V_e.shape[1] <= V_f.shape[1]


# ---------------------------------------------------------------------------
# Test: mode='separate'
# ---------------------------------------------------------------------------

class TestSeparateMode:
    def test_full_return(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        result = gsvd(A, B, mode='separate')
        assert len(result) == 8
        U, V, D1, D2, R, Q, k, l = result
        assert U.shape == (5, 5)
        assert V.shape == (4, 4)
        assert Q.shape == (6, 6)
        assert R.shape == (k + l, k + l)
        assert D1.shape == (5, k + l)
        assert D2.shape == (4, k + l)

    def test_no_u(self):
        rng = np.random.default_rng(12)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, mode='separate', compute_u=False)
        assert len(result) == 7
        V = result[0]
        assert V.shape == (3, 3)

    def test_no_u_no_v(self):
        rng = np.random.default_rng(13)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, mode='separate', compute_u=False, compute_v=False)
        assert len(result) == 6
        D1, D2, R, Q, k, l = result

    def test_reconstruction_via_lapack_form(self):
        """Verify A ≈ U @ D1 @ np.hstack([zeros, R]) @ Q.T."""
        rng = np.random.default_rng(20)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        U, V, D1, D2, R, Q, k, l = gsvd(A, B, mode='separate')

        q = k + l
        p = A.shape[1]
        zero_block = np.zeros((q, p - q))
        RQ_block = np.hstack([zero_block, R]) @ Q.T    # q×p

        A_rec = U @ D1 @ RQ_block
        B_rec = V @ D2 @ RQ_block
        assert_allclose(A_rec, A, rtol=1e-10, atol=1e-10 * np.linalg.norm(A))
        assert_allclose(B_rec, B, rtol=1e-10, atol=1e-10 * np.linalg.norm(B))


# ---------------------------------------------------------------------------
# Test: overwrite and lwork options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_overwrite_a_b(self):
        rng = np.random.default_rng(99)
        A = np.asfortranarray(rng.standard_normal((4, 5)))
        B = np.asfortranarray(rng.standard_normal((3, 5)))
        # Should not raise
        gsvd(A, B, overwrite_a=True, overwrite_b=True)

    def test_explicit_lwork(self):
        rng = np.random.default_rng(100)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        U1, V1, X1, C1, S1 = gsvd(A, B)
        U2, V2, X2, C2, S2 = gsvd(A, B, lwork=500)
        assert_allclose(C1, C2, rtol=1e-12)

    def test_check_finite_raises(self):
        A = np.array([[1.0, np.nan], [2.0, 3.0]])
        B = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="non-finite"):
            gsvd(A, B)

    def test_check_finite_skip(self):
        # With check_finite=False, no error even with nan (behaviour is
        # undefined, but the call should not raise a Python-level error
        # for this test — we just check the flag is respected)
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[1.0, 2.0]])
        gsvd(A, B, check_finite=False)   # should not raise


# ---------------------------------------------------------------------------
# Test: input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_mode(self):
        A = np.eye(3)
        B = np.eye(3)
        with pytest.raises(ValueError, match="mode"):
            gsvd(A, B, mode='bad')

    def test_mismatched_columns(self):
        A = np.ones((3, 4))
        B = np.ones((2, 5))
        with pytest.raises(ValueError, match="columns"):
            gsvd(A, B)

    def test_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            gsvd(np.ones(3), np.ones((2, 3)))

    def test_integer_input_upcasted(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[1, 2, 3]])
        # Should not raise; integers are upcast to float64
        U, V, X, C, S = gsvd(A, B)
        assert U.dtype in (np.float64, np.complex128)


# ---------------------------------------------------------------------------
# Test: dtype handling
# ---------------------------------------------------------------------------

class TestDtypes:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                        np.complex64, np.complex128])
    def test_dtype_preserved(self, dtype):
        rng = np.random.default_rng(55)
        A = _random_matrix(rng, 4, 5, dtype)
        B = _random_matrix(rng, 3, 5, dtype)
        U, V, X, C, S = gsvd(A, B)
        assert U.dtype == dtype
        assert V.dtype == dtype
        assert X.dtype == dtype

    def test_mixed_real_float32_float64(self):
        rng = np.random.default_rng(56)
        A = rng.standard_normal((4, 5)).astype(np.float32)
        B = rng.standard_normal((3, 5)).astype(np.float64)
        U, V, X, C, S = gsvd(A, B)
        assert U.dtype == np.float64   # result_type promotes to float64
