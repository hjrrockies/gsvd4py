"""
Core implementation of gsvd() using ctypes LAPACK calls.

LAPACK vs spec notation mapping
--------------------------------
The spec uses Matlab-style dimensions: A is m×p, B is n×p.
LAPACK dggsvd3 uses: A is M×N, B is P×N.

  spec m  →  LAPACK M   (rows of A)
  spec n  →  LAPACK P   (rows of B)
  spec p  →  LAPACK N   (columns, shared)

LAPACK decomposition (real case)
----------------------------------
  A = U * D1 * [0, R] * Q^T
  B = V * D2 * [0, R] * Q^T

where:
  U   M×M orthogonal   (spec: m×m)
  V   P×P orthogonal   (spec: n×n)
  Q   N×N orthogonal   (spec: p×p)
  D1  M×q "diagonal"   (spec: m×q)   q = K+L
  D2  P×q "diagonal"   (spec: n×q)
  R   q×q upper-triangular, stored inside A (and B if M < q)
  [0, R]  q×N block matrix

Matlab-style X (full / econ modes)
------------------------------------
  Q2     = Q[:, p-q:]        last q columns of Q,  shape p×q
  X      = Q2 @ conj(R).T    shape p×q
  then A = U * C * X^H,  B = V * S * X^H

D1 / D2 structure (ALPHA, BETA from LAPACK)
--------------------------------------------
Case m >= q  (M >= K+L):
  ALPHA[0:k]   = 1,  BETA[0:k]   = 0      (infinite GSVs)
  ALPHA[k:k+l] = C,  BETA[k:k+l] = S      (finite GSVs, C²+S²=I)
  ALPHA[k+l:p] = 0,  BETA[k+l:p] = 0

Case m < q  (M < K+L, still K <= M):
  ALPHA[0:k]   = 1,  BETA[0:k]   = 0
  ALPHA[k:m]   = C,  BETA[k:m]   = S      (first M-K pairs)
  ALPHA[m:q]   = 0,  BETA[m:q]   = 1      (identity block in D2)
  ALPHA[q:p]   = 0,  BETA[q:p]   = 0
"""

import ctypes

import numpy as np

from ._lapack import get_ggsvd3

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    np.dtype('float32'):    ('s', np.dtype('float32'),   False),
    np.dtype('float64'):    ('d', np.dtype('float64'),   False),
    np.dtype('complex64'):  ('c', np.dtype('float32'),   True),
    np.dtype('complex128'): ('z', np.dtype('float64'),   True),
}


def _resolve_dtype(a, b):
    """Return (lapack_dtype, real_dtype, is_complex) for inputs a and b."""
    dtype = np.result_type(a, b)
    # Upcast integers / booleans to float64
    if not (np.issubdtype(dtype, np.floating) or
            np.issubdtype(dtype, np.complexfloating)):
        dtype = np.float64
    # Upcast float16 → float32, etc.
    if dtype == np.float16:
        dtype = np.float32
    return _DTYPE_MAP[np.dtype(dtype)]


# ---------------------------------------------------------------------------
# ctypes helpers
# ---------------------------------------------------------------------------

_c_int_p  = ctypes.POINTER(ctypes.c_int)
_c_void_p = ctypes.c_void_p


def _ptr(arr):
    """Return a c_void_p pointing to arr's data buffer."""
    return arr.ctypes.data_as(_c_void_p)


def _iptr(val):
    """Return a ctypes pointer to a c_int value."""
    return ctypes.byref(ctypes.c_int(val))


# ---------------------------------------------------------------------------
# LAPACK call wrapper
# ---------------------------------------------------------------------------

def _call_ggsvd3(a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
                 jobu, jobv, lwork, fn, is_complex, real_dtype,
                 uses_hidden_lengths):
    """Call ?ggsvd3 once (workspace query or actual computation).

    All array arguments must already be Fortran-contiguous and correctly typed.
    Returns (k, l, info).
    """
    m_lap, p_lap = a_f.shape   # LAPACK M, N
    n_lap        = b_f.shape[0]  # LAPACK P
    q_lap        = p_lap         # = LAPACK N, used for LDQ

    lwork_val = lwork if lwork is not None else 1
    work = np.zeros(max(1, lwork_val), dtype=a_f.dtype)
    lwork_c = ctypes.c_int(-1 if lwork is None else lwork_val)

    k_c    = ctypes.c_int(0)
    l_c    = ctypes.c_int(0)
    info_c = ctypes.c_int(0)

    # Dummy 1×1 arrays for when U or V is not computed
    dummy = np.zeros((1, 1), dtype=a_f.dtype, order='F')

    u_ptr   = _ptr(u_f)   if u_f is not None else _ptr(dummy)
    v_ptr   = _ptr(v_f)   if v_f is not None else _ptr(dummy)
    ldu_val = m_lap        if u_f is not None else 1
    ldv_val = n_lap        if v_f is not None else 1

    # jobu / jobv chars (single byte)
    jobu_b = jobu.encode()
    jobv_b = jobv.encode()
    jobq_b = b'Q'

    args = [
        jobu_b, jobv_b, jobq_b,
        _iptr(m_lap), _iptr(p_lap), _iptr(n_lap),
        ctypes.byref(k_c), ctypes.byref(l_c),
        _ptr(a_f),   _iptr(m_lap),
        _ptr(b_f),   _iptr(n_lap),
        _ptr(alpha), _ptr(beta),
        u_ptr,       _iptr(ldu_val),
        v_ptr,       _iptr(ldv_val),
        _ptr(q_f),   _iptr(q_lap),
        _ptr(work),  ctypes.byref(lwork_c),
    ]

    if is_complex:
        rwork = np.zeros(2 * p_lap, dtype=real_dtype)
        args += [_ptr(rwork)]

    args += [_ptr(iwork), ctypes.byref(info_c)]

    if uses_hidden_lengths:
        one = ctypes.c_size_t(1)
        args += [one, one, one]

    fn(*args)

    if lwork is None:
        # workspace query: return optimal lwork from work[0]
        return int(work[0].real)

    return k_c.value, l_c.value, info_c.value


# ---------------------------------------------------------------------------
# Output construction helpers
# ---------------------------------------------------------------------------

def _build_C_S(alpha, beta, m, n, k, l):
    """Build dense C (m×q) and S (n×q) from LAPACK ALPHA / BETA vectors.

    Returns (C, S) as float64 arrays regardless of dtype (GSVs are real).
    """
    q = k + l
    C = np.zeros((m, q))
    S = np.zeros((n, q))

    if k > 0:
        C[:k, :k] = np.eye(k)   # identity block

    if m >= q:                   # Case 1: M >= K+L
        idx = np.arange(l)
        C[k + idx, k + idx] = alpha[k:k+l]
        S[idx,     k + idx] = beta[k:k+l]
    else:                        # Case 2: M < K+L  (still K <= M)
        mk = m - k               # number of (cos, sin) pairs that fit in D1
        if mk > 0:
            idx = np.arange(mk)
            C[k + idx, k + idx] = alpha[k:m]
            S[idx,     k + idx] = beta[k:m]
        kl_m = q - m             # K+L-M  = size of identity block in D2
        if kl_m > 0:
            idx2 = np.arange(kl_m)
            S[mk + idx2, m + idx2] = 1.0

    return C, S


def _extract_R(a_f, b_f, m, n, p, k, l):
    """Extract the (k+l)×(k+l) upper-triangular R from the modified A (and B).

    LAPACK stores R in A[0:k+l, p-k-l:p] (0-indexed, Fortran-order array).
    If m < k+l, the bottom k+l-m rows come from B[0:k+l-m, p-k-l:p].
    """
    q = k + l
    R = np.zeros((q, q), dtype=a_f.dtype)
    col_start = p - q

    if m >= q:
        R[:] = a_f[:q, col_start:]
    else:
        kl_m = q - m                    # K+L-M rows of R that overflow into B
        R[:m, :] = a_f[:m, col_start:]
        # LAPACK stores R[m:q, m:q] (upper-triangular block) in
        # B(M-K+1 : L,  N+M-K-L+1 : N)  [Fortran 1-indexed]
        # = b_f[m-k : l,  col_start+m : p]  [Python 0-indexed]
        b_row = m - k
        b_col = col_start + m           # = p - kl_m
        R[m:, m:] = b_f[b_row:b_row + kl_m, b_col:]

    return R


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gsvd(a, b, mode='full', compute_u=True, compute_v=True,
         overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True):
    """Generalized Singular Value Decomposition.

    Computes the GSVD of the matrix pair (a, b) using the LAPACK routine
    ?ggsvd3 linked via the same LAPACK library as SciPy.

    Parameters
    ----------
    a : (m, p) array_like
    b : (n, p) array_like
    mode : {'full', 'econ', 'separate'}, default 'full'
        'full'     — Full Matlab-style: U (m×m), V (n×n), X (p×q), C (m×q),
                     S (n×q), where q = k+l is the numerical rank of [a; b].
        'econ'     — Economy Matlab-style: U (m×min(m,q)), V (n×min(n,q)),
                     X (p×q), C (min(m,q)×q), S (min(n,q)×q).
        'separate' — Raw LAPACK output (no rank truncation): U, V, D1, D2,
                     R, Q, k, l.
    compute_u : bool, default True
        Compute left singular vectors of a.
    compute_v : bool, default True
        Compute left singular vectors of b.
    overwrite_a : bool, default False
        Allow overwriting a (avoids a copy if True and a is already
        Fortran-contiguous with the correct dtype).
    overwrite_b : bool, default False
        Allow overwriting b (same as overwrite_a).
    lwork : int or None, default None
        LAPACK work array size.  None (or -1) triggers an optimal query.
    check_finite : bool, default True
        Check that a and b contain only finite values.

    Returns
    -------
    mode='full' or 'econ':
        If compute_u and compute_v:     U, V, X, C, S
        If compute_u and not compute_v: U, X, C, S
        If not compute_u and compute_v: V, X, C, S
        If not compute_u and compute_v: X, C, S

    mode='separate':
        If compute_u and compute_v:     U, V, D1, D2, R, Q, k, l
        If compute_u and not compute_v: U, D1, D2, R, Q, k, l
        If not compute_u and compute_v: V, D1, D2, R, Q, k, l
        If not compute_u and compute_v: D1, D2, R, Q, k, l
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if mode not in ('full', 'econ', 'separate'):
        raise ValueError(f"mode must be 'full', 'econ', or 'separate', got {mode!r}")

    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim != 2:
        raise ValueError(f"a must be 2-D, got shape {a.shape}")
    if b.ndim != 2:
        raise ValueError(f"b must be 2-D, got shape {b.shape}")

    m, p = a.shape
    n    = b.shape[0]
    if b.shape[1] != p:
        raise ValueError(
            f"a and b must have the same number of columns: "
            f"{p} != {b.shape[1]}"
        )

    if check_finite:
        if not np.all(np.isfinite(a)):
            raise ValueError("Array a contains non-finite values.")
        if not np.all(np.isfinite(b)):
            raise ValueError("Array b contains non-finite values.")

    # ------------------------------------------------------------------
    # Dtype resolution + array preparation
    # ------------------------------------------------------------------
    dtype_char, real_dtype, is_complex = _resolve_dtype(a, b)
    dtype = np.dtype('complex64' if dtype_char == 'c'
                     else 'complex128' if dtype_char == 'z'
                     else 'float32' if dtype_char == 's'
                     else 'float64')

    def _prep(arr, overwrite):
        if overwrite and arr.dtype == dtype and np.isfortran(arr):
            return arr
        return np.array(arr, dtype=dtype, order='F', copy=True)

    a_f = _prep(a, overwrite_a)
    b_f = _prep(b, overwrite_b)

    # ------------------------------------------------------------------
    # Load LAPACK function
    # ------------------------------------------------------------------
    fn, uses_hidden_lengths = get_ggsvd3(dtype_char)

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    jobu_char = 'U' if compute_u else 'N'
    jobv_char = 'V' if compute_v else 'N'

    alpha  = np.zeros(p, dtype=real_dtype)
    beta   = np.zeros(p, dtype=real_dtype)
    iwork  = np.zeros(p, dtype=np.int32)
    q_f    = np.zeros((p, p), dtype=dtype, order='F')
    u_f    = np.zeros((m, m), dtype=dtype, order='F') if compute_u else None
    v_f    = np.zeros((n, n), dtype=dtype, order='F') if compute_v else None

    # ------------------------------------------------------------------
    # Workspace query
    # ------------------------------------------------------------------
    if lwork is None or lwork == -1:
        opt = _call_ggsvd3(
            a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
            jobu_char, jobv_char,
            lwork=None, fn=fn, is_complex=is_complex,
            real_dtype=real_dtype, uses_hidden_lengths=uses_hidden_lengths,
        )
        lwork_use = max(opt, 1)
    else:
        lwork_use = lwork

    # ------------------------------------------------------------------
    # Actual LAPACK call
    # ------------------------------------------------------------------
    k, l, info = _call_ggsvd3(
        a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
        jobu_char, jobv_char,
        lwork=lwork_use, fn=fn, is_complex=is_complex,
        real_dtype=real_dtype, uses_hidden_lengths=uses_hidden_lengths,
    )

    if info < 0:
        raise ValueError(f"Illegal argument #{-info} passed to dggsvd3.")
    if info > 0:
        raise np.linalg.LinAlgError(
            f"LAPACK ?ggsvd3 failed to converge (info={info})."
        )

    q_rank = k + l   # effective numerical rank

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    if mode == 'separate':
        return _build_separate(
            a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
            m, n, p, k, l, q_rank,
            compute_u, compute_v, real_dtype,
        )
    else:
        return _build_matlab_style(
            a_f, b_f, alpha, beta, u_f, v_f, q_f,
            m, n, p, k, l, q_rank,
            mode, compute_u, compute_v,
        )


# ---------------------------------------------------------------------------
# Post-processing: separate mode
# ---------------------------------------------------------------------------

def _build_separate(a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
                    m, n, p, k, l, q_rank,
                    compute_u, compute_v, real_dtype):
    R = _extract_R(a_f, b_f, m, n, p, k, l)
    D1, D2 = _build_C_S(alpha, beta, m, n, k, l)

    # Convert to C-order for return
    R  = np.ascontiguousarray(R)
    D1 = np.ascontiguousarray(D1)
    D2 = np.ascontiguousarray(D2)
    Q  = np.ascontiguousarray(q_f)

    result = []
    if compute_u:
        result.append(np.ascontiguousarray(u_f))
    if compute_v:
        result.append(np.ascontiguousarray(v_f))
    result += [D1, D2, R, Q, k, l]
    return tuple(result)


# ---------------------------------------------------------------------------
# Post-processing: full / econ modes
# ---------------------------------------------------------------------------

def _build_matlab_style(a_f, b_f, alpha, beta, u_f, v_f, q_f,
                        m, n, p, k, l, q_rank,
                        mode, compute_u, compute_v):
    # Build C and S (real-valued diagonal matrices)
    C_full, S_full = _build_C_S(alpha, beta, m, n, k, l)

    # Extract R then build X = Q2 @ conj(R).T
    R   = _extract_R(a_f, b_f, m, n, p, k, l)
    Q2  = np.asarray(q_f)[:, p - q_rank:]    # p×q_rank
    X   = Q2 @ np.conj(R).T                  # p×q_rank

    # Full mode: U is m×m, V is n×n, C is m×q, S is n×q
    # Econ mode: truncate U to m×r, V to n×r, C to r×q, S to r×q
    #   where r = min(m, q_rank) for U/C  and  min(n, q_rank) for V/S
    if mode == 'full':
        C = C_full
        S = S_full
        U_out = np.ascontiguousarray(u_f) if compute_u else None
        V_out = np.ascontiguousarray(v_f) if compute_v else None
    else:  # 'econ'
        ru = min(m, q_rank)
        rv = min(n, q_rank)
        C  = np.ascontiguousarray(C_full[:ru, :])
        S  = np.ascontiguousarray(S_full[:rv, :])
        U_out = np.ascontiguousarray(u_f[:, :ru]) if compute_u else None
        V_out = np.ascontiguousarray(v_f[:, :rv]) if compute_v else None

    X = np.ascontiguousarray(X)
    C = np.ascontiguousarray(C)
    S = np.ascontiguousarray(S)

    result = []
    if compute_u:
        result.append(U_out)
    if compute_v:
        result.append(V_out)
    result += [X, C, S]
    return tuple(result)
