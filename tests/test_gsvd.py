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
  - Large matrices (p ~ 100)
  - Rank-deficient pairs (q < p, A individually rank-deficient)
  - Pairs with infinite generalized singular values (k > 0)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gsvd4py import gsvd, gsvdvals
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


def _check_reconstruction(U, V, C, S, X, A, B, rtol):
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

        U, V, C, S, X = gsvd(A, B, mode='full')

        assert U.shape == (m, m)
        assert V.shape == (n, n)
        assert X.shape[0] == p
        assert C.shape[0] == m
        assert S.shape[0] == n
        assert C.shape[1] == X.shape[1] == S.shape[1]  # same q

        _check_reconstruction(U, V, C, S, X, A, B, rtol)
        _check_unitary(U, rtol)
        _check_unitary(V, rtol)

    def test_no_u_no_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=False, compute_v=False)
        assert len(result) == 3
        C, S, X = result
        assert X.shape[0] == 5

    def test_no_u_with_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=False, compute_v=True)
        assert len(result) == 4
        V, C, S, X = result
        assert V.shape == (3, 3)

    def test_with_u_no_v(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=True, compute_v=False)
        assert len(result) == 4
        U, C, S, X = result
        assert U.shape == (4, 4)

    def test_c_diagonal_descending(self):
        """Diagonal of C must be in non-increasing order."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        U, V, C, S, X = gsvd(A, B)
        q = C.shape[1]
        c_diag = np.array([C[j, j] for j in range(min(A.shape[0], q))])
        assert np.all(c_diag[:-1] >= c_diag[1:] - 1e-14), \
            f"C diagonal not descending: {c_diag}"

    def test_compute_right_false_full(self):
        """compute_right=False drops X from the return tuple."""
        rng = np.random.default_rng(5)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_right=False)
        assert len(result) == 4
        U, V, C, S = result
        assert U.shape == (4, 4)
        assert V.shape == (3, 3)

    def test_compute_right_false_no_uv(self):
        """compute_right=False with no U/V returns only (C, S)."""
        rng = np.random.default_rng(6)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((3, 5))
        result = gsvd(A, B, compute_u=False, compute_v=False, compute_right=False)
        assert len(result) == 2
        C, S = result
        assert C.shape[0] == 4   # m rows

    def test_compute_right_false_matches_full(self):
        """C and S from compute_right=False match those from a full call."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        U, V, C_full, S_full, X = gsvd(A, B)
        U2, V2, C_nr, S_nr = gsvd(A, B, compute_right=False)
        assert_allclose(C_nr, C_full, rtol=1e-12)
        assert_allclose(S_nr, S_full, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test: sorting of C and S diagonals
# ---------------------------------------------------------------------------

class TestSorting:
    """Verify that the diagonal of C is non-increasing and the corresponding
    entries of S are non-decreasing, across modes and edge cases."""

    @staticmethod
    def _extract_cs(C, S):
        """Column-wise max gives the GSV pairs (same convention as gsvdvals)."""
        return np.max(C, axis=0), np.max(S, axis=0)

    @pytest.mark.parametrize("mode", ['full', 'econ'])
    @pytest.mark.parametrize("shape", [
        (5, 4, 6),   # generic
        (4, 4, 4),   # square
        (3, 5, 6),   # m < p
        (6, 3, 4),   # m > p, n < p
        (2, 4, 4),   # m < n = p  (likely m < q, triggers identity block in S)
    ])
    def test_cs_sorted(self, shape, mode):
        """C diagonal non-increasing; S finite block non-decreasing."""
        m, n, p = shape
        rng = np.random.default_rng(42)
        A = rng.standard_normal((m, p))
        B = rng.standard_normal((n, p))

        C, S, X = gsvd(A, B, mode=mode, compute_u=False, compute_v=False)
        c, s = self._extract_cs(C, S)

        assert np.all(c[:-1] >= c[1:]), \
            f"C not non-increasing ({mode} {shape}): {c}"
        assert np.all(s[:-1] <= s[1:]), \
            f"S not non-decreasing ({mode} {shape}): {s}"

    def test_k_gt_0(self):
        """When k > 0 (infinite GSVs), ones in C come first; finite block sorted."""
        # B = [1, 0, 0] has null space span{e2, e3}; for generic A with 3 columns
        # both null-space directions are in col(A), so k = 2, l = 1.
        rng = np.random.default_rng(77)
        A = rng.standard_normal((3, 3))
        B = np.array([[1.0, 0.0, 0.0]])

        U, V, C, S, X = gsvd(A, B)
        c, s = self._extract_cs(C, S)

        k = int(np.sum(np.isclose(c, 1.0)))
        assert k >= 1, f"Expected k >= 1 (B has 2-D null space), got k={k}"
        assert np.all(c[:-1] >= c[1:]), f"C not non-increasing: {c}"
        assert np.all(s[:-1] <= s[1:]), f"S not non-decreasing: {s}"

    def test_m_lt_q(self):
        """When m < q, the identity block (s=1) in S follows the sorted finite block."""
        # A is 2×4, B is 4×4 full-rank → k=0, q=4, m=2 < q.
        # S should be [sin_asc, 1, 1] and C should be [cos_desc, 0, 0].
        rng = np.random.default_rng(88)
        A = rng.standard_normal((2, 4))
        B = rng.standard_normal((4, 4))

        U, V, C, S, X = gsvd(A, B)
        q = C.shape[1]
        assert A.shape[0] < q, "Test setup failed: expected m < q"

        c, s = self._extract_cs(C, S)
        assert np.all(c[:-1] >= c[1:]), f"C not non-increasing (m<q): {c}"
        assert np.all(s[:-1] <= s[1:]), f"S not non-decreasing (m<q): {s}"


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

        U, V, C, S, X = gsvd(A, B, mode='econ')
        q = X.shape[1]

        assert U.shape == (m, min(m, q))
        assert V.shape == (n, min(n, q))
        assert C.shape == (min(m, q), q)
        assert S.shape == (min(n, q), q)

        _check_reconstruction(U, V, C, S, X, A, B, rtol)

    def test_econ_smaller_than_full(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((8, 5))
        B = rng.standard_normal((6, 5))
        U_f, V_f, C_f, S_f, X_f = gsvd(A, B, mode='full')
        U_e, V_e, C_e, S_e, X_e = gsvd(A, B, mode='econ')
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

    def test_compute_right_false_drops_q(self):
        """compute_right=False drops Q; R is still returned."""
        rng = np.random.default_rng(14)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        result = gsvd(A, B, mode='separate', compute_right=False)
        assert len(result) == 7          # U, V, D1, D2, R, k, l  (no Q)
        U, V, D1, D2, R, k, l = result
        assert R.shape == (k + l, k + l)

    def test_compute_right_false_r_matches(self):
        """R from compute_right=False matches R from a full separate call."""
        rng = np.random.default_rng(15)
        A = rng.standard_normal((5, 6))
        B = rng.standard_normal((4, 6))
        U, V, D1, D2, R_full, Q, k, l = gsvd(A, B, mode='separate')
        U2, V2, D1_2, D2_2, R_nr, k2, l2 = gsvd(A, B, mode='separate',
                                                  compute_right=False)
        assert_allclose(R_nr, R_full, rtol=1e-12)

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
        U1, V1, C1, S1, X1 = gsvd(A, B)
        U2, V2, C2, S2, X2 = gsvd(A, B, lwork=500)
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
        U, V, C, S, X = gsvd(A, B)
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
        U, V, C, S, X = gsvd(A, B)
        assert U.dtype == dtype
        assert V.dtype == dtype
        assert X.dtype == dtype

    def test_mixed_real_float32_float64(self):
        rng = np.random.default_rng(56)
        A = rng.standard_normal((4, 5)).astype(np.float32)
        B = rng.standard_normal((3, 5)).astype(np.float64)
        U, V, C, S, X = gsvd(A, B)
        assert U.dtype == np.float64   # result_type promotes to float64


# ---------------------------------------------------------------------------
# Test: large matrices
# ---------------------------------------------------------------------------

class TestLargeMatrices:
    def test_reconstruction_large(self):
        """Reconstruction holds for large matrices (p=100)."""
        rng = np.random.default_rng(42)
        m, n, p = 200, 150, 100
        A = rng.standard_normal((m, p))
        B = rng.standard_normal((n, p))
        U, V, C, S, X = gsvd(A, B)
        _check_reconstruction(U, V, C, S, X, A, B, rtol=1e-10)

    def test_sorting_large(self):
        """c non-increasing and s non-decreasing when m, n >> p.

        Regression test for the lexsort tie-breaking fix: with m, n >> p,
        LAPACK sets several alpha values to exactly 1.0 while others are
        very close but not exactly 1.0.  A naive stable argsort of -c leaves
        the corresponding s values unsorted within the exact-1.0 tied group.
        """
        rng = np.random.default_rng(43)
        m, n, p = 200, 150, 20   # m, n >> p: many cosines cluster near 1
        A = rng.standard_normal((m, p))
        B = rng.standard_normal((n, p))
        c, s = gsvdvals(A, B)
        assert np.all(c[:-1] >= c[1:]), f"c not non-increasing: {c}"
        assert np.all(s[:-1] <= s[1:]), f"s not non-decreasing: {s}"


# ---------------------------------------------------------------------------
# Test: rank-deficient matrix pairs
# ---------------------------------------------------------------------------

class TestRankDeficient:
    def test_stacked_rank_deficient(self):
        """When [A;B] has numerical rank r < p, q = k+l = r and A = UCX^H."""
        rng = np.random.default_rng(42)
        m, n, p, r = 8, 6, 10, 4
        # Force A and B into the same r-dimensional column space so that
        # [A; B] has rank r and the GSVD numerical rank q = r.
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        V = V[:, :r]                               # p×r, orthonormal
        A = rng.standard_normal((m, r)) @ V.T      # m×p, rank r
        B = rng.standard_normal((n, r)) @ V.T      # n×p, rank r

        U, V_out, C, S, X = gsvd(A, B)
        q = C.shape[1]
        assert q == r, f"Expected numerical rank q={r}, got q={q}"
        _check_reconstruction(U, V_out, C, S, X, A, B, rtol=1e-10)

    def test_a_rank_deficient(self):
        """A individually rank-deficient (rank r < p) while [A;B] has full rank."""
        rng = np.random.default_rng(43)
        m, n, p, r = 8, 5, 6, 3
        # A = (m×r)(r×p) has rank r; generic B is full-rank so [A;B] has rank p.
        A = rng.standard_normal((m, r)) @ rng.standard_normal((r, p))
        B = rng.standard_normal((n, p))

        U, V, C, S, X = gsvd(A, B)
        _check_reconstruction(U, V, C, S, X, A, B, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test: infinite generalized singular values (k > 0)
# ---------------------------------------------------------------------------

class TestInfiniteGSVs:
    """Tests for pairs with k > 0 (infinite generalized singular values).

    Infinite GSVs arise when a direction lies in col(A) but in null(B).
    Construction: take m >= p (so col(A) = R^p for generic A) and n < p
    (so null(B) is non-trivial).  For generic random matrices, k = p - n
    and l = n exactly.
    """

    @staticmethod
    def _make_pair(rng, m, n, p):
        assert m >= p and n < p
        return rng.standard_normal((m, p)), rng.standard_normal((n, p))

    def test_k_value(self):
        """k = p - n and l = n when A has full column rank and n < p."""
        rng = np.random.default_rng(99)
        m, n, p = 8, 3, 7
        A, B = self._make_pair(rng, m, n, p)

        *_, k, l = gsvd(A, B, mode='separate',
                        compute_u=False, compute_v=False, compute_right=False)
        assert k == p - n, f"Expected k={p - n}, got k={k}"
        assert l == n,     f"Expected l={n}, got l={l}"

    def test_reconstruction_with_k(self):
        """Reconstruction A = U@C@X^H and B = V@S@X^H hold when k > 0."""
        rng = np.random.default_rng(100)
        m, n, p = 8, 3, 7
        A, B = self._make_pair(rng, m, n, p)

        U, V, C, S, X = gsvd(A, B)
        _check_reconstruction(U, V, C, S, X, A, B, rtol=1e-10)

    def test_c_ones_s_zeros_in_k_block(self):
        """The first k entries of c equal 1 and the corresponding s entries equal 0."""
        rng = np.random.default_rng(101)
        m, n, p = 10, 4, 8   # k = p - n = 4
        A, B = self._make_pair(rng, m, n, p)
        k_expected = p - n

        c, s = gsvdvals(A, B)
        assert_allclose(c[:k_expected], np.ones(k_expected),  atol=1e-12)
        assert_allclose(s[:k_expected], np.zeros(k_expected), atol=1e-12)
