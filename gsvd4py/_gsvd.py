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

from ._lapack import get_ggsvd3, get_ggsvp3, get_tgsja

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
                 jobu, jobv, jobq, lwork, fn, is_complex, real_dtype,
                 uses_hidden_lengths):
    """Call ?ggsvd3 once (workspace query or actual computation).

    All array arguments must already be Fortran-contiguous and correctly typed.
    Returns (k, l, info).
    """
    m_lap, p_lap = a_f.shape   # LAPACK M, N
    n_lap        = b_f.shape[0]  # LAPACK P

    lwork_val = lwork if lwork is not None else 1
    work = np.zeros(max(1, lwork_val), dtype=a_f.dtype)
    lwork_c = ctypes.c_int(-1 if lwork is None else lwork_val)

    k_c    = ctypes.c_int(0)
    l_c    = ctypes.c_int(0)
    info_c = ctypes.c_int(0)

    # Dummy 1×1 arrays for when U, V, or Q is not computed
    dummy = np.zeros((1, 1), dtype=a_f.dtype, order='F')

    u_ptr   = _ptr(u_f)   if u_f is not None else _ptr(dummy)
    v_ptr   = _ptr(v_f)   if v_f is not None else _ptr(dummy)
    q_ptr   = _ptr(q_f)   if q_f is not None else _ptr(dummy)
    ldu_val = m_lap        if u_f is not None else 1
    ldv_val = n_lap        if v_f is not None else 1
    ldq_val = p_lap        if q_f is not None else 1

    # jobu / jobv / jobq chars (single byte)
    jobu_b = jobu.encode()
    jobv_b = jobv.encode()
    jobq_b = jobq.encode()

    args = [
        jobu_b, jobv_b, jobq_b,
        _iptr(m_lap), _iptr(p_lap), _iptr(n_lap),
        ctypes.byref(k_c), ctypes.byref(l_c),
        _ptr(a_f),   _iptr(m_lap),
        _ptr(b_f),   _iptr(n_lap),
        _ptr(alpha), _ptr(beta),
        u_ptr,       _iptr(ldu_val),
        v_ptr,       _iptr(ldv_val),
        q_ptr,       _iptr(ldq_val),
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
# tola / tolb helpers
# ---------------------------------------------------------------------------

def _default_tola_tolb(a_f, b_f, real_dtype):
    """Return LAPACK-default TOLA and TOLB.

    Matches the internal formula used by dggsvd3:
      anorm = dlange('1', ...)   -- 1-norm (max absolute column sum)
      tola  = max(M, N) * max(anorm, unfl) * ulp
    where ulp = machine epsilon and unfl = safe minimum (DLAMCH 'Safe Minimum').
    """
    m_lap, p_lap = a_f.shape    # LAPACK M, N
    n_lap        = b_f.shape[0] # LAPACK P
    fi   = np.finfo(real_dtype)
    eps  = fi.eps
    unfl = fi.tiny
    anorm = np.linalg.norm(a_f, ord=1)
    bnorm = np.linalg.norm(b_f, ord=1)
    tola = max(m_lap, p_lap) * max(float(anorm), unfl) * eps
    tolb = max(n_lap, p_lap) * max(float(bnorm), unfl) * eps
    return tola, tolb


def _call_ggsvp3(a_f, b_f, u_f, v_f, q_f, iwork,
                 jobu, jobv, jobq, tola, tolb, lwork,
                 fn, is_complex, real_dtype, uses_hidden_lengths):
    """Call ?ggsvp3 once (workspace query or actual computation).

    LAPACK signature (real):
      DGGSVP3(JOBU, JOBV, JOBQ, M, P, N, A, LDA, B, LDB, TOLA, TOLB,
              K, L, U, LDU, V, LDV, Q, LDQ, IWORK, TAU, WORK, LWORK, INFO)

    Complex variants add RWORK (size 2*N) after LWORK and before INFO.
    gfortran ABI appends three size_t(1) hidden args after INFO.

    lwork=None triggers workspace query; returns optimal lwork as int.
    Actual call returns (k, l, info).
    """
    m_lap, p_lap = a_f.shape    # LAPACK M, N
    n_lap        = b_f.shape[0] # LAPACK P

    lwork_val = lwork if lwork is not None else 1
    work = np.zeros(max(1, lwork_val), dtype=a_f.dtype)
    lwork_c = ctypes.c_int(-1 if lwork is None else lwork_val)

    k_c    = ctypes.c_int(0)
    l_c    = ctypes.c_int(0)
    info_c = ctypes.c_int(0)

    dummy = np.zeros((1, 1), dtype=a_f.dtype, order='F')

    u_ptr   = _ptr(u_f)   if u_f is not None else _ptr(dummy)
    v_ptr   = _ptr(v_f)   if v_f is not None else _ptr(dummy)
    q_ptr   = _ptr(q_f)   if q_f is not None else _ptr(dummy)
    ldu_val = m_lap        if u_f is not None else 1
    ldv_val = n_lap        if v_f is not None else 1
    ldq_val = p_lap        if q_f is not None else 1

    jobu_b = jobu.encode()
    jobv_b = jobv.encode()
    jobq_b = jobq.encode()

    tola_c = ctypes.c_double(tola) if real_dtype == np.float64 else ctypes.c_float(tola)
    tolb_c = ctypes.c_double(tolb) if real_dtype == np.float64 else ctypes.c_float(tolb)

    tau = np.zeros(p_lap, dtype=a_f.dtype)

    args = [
        jobu_b, jobv_b, jobq_b,
        _iptr(m_lap), _iptr(n_lap), _iptr(p_lap),
        _ptr(a_f),   _iptr(m_lap),
        _ptr(b_f),   _iptr(n_lap),
        ctypes.byref(tola_c), ctypes.byref(tolb_c),
        ctypes.byref(k_c), ctypes.byref(l_c),
        u_ptr,       _iptr(ldu_val),
        v_ptr,       _iptr(ldv_val),
        q_ptr,       _iptr(ldq_val),
        _ptr(iwork), _ptr(tau),
    ]

    if is_complex:
        rwork = np.zeros(2 * p_lap, dtype=real_dtype)
        if uses_hidden_lengths:
            # Standard gfortran ABI: WORK, LWORK, RWORK, INFO
            args += [_ptr(work), ctypes.byref(lwork_c), _ptr(rwork)]
        else:
            # Accelerate NEWLAPACK: WORK, RWORK, LWORK, INFO (RWORK before LWORK)
            args += [_ptr(work), _ptr(rwork), ctypes.byref(lwork_c)]
    else:
        args += [_ptr(work), ctypes.byref(lwork_c)]

    args += [ctypes.byref(info_c)]

    if uses_hidden_lengths:
        one = ctypes.c_size_t(1)
        args += [one, one, one]

    fn(*args)

    if lwork is None:
        return int(work[0].real)

    return k_c.value, l_c.value, info_c.value


def _call_tgsja(a_f, b_f, alpha, beta, u_f, v_f, q_f,
                jobu, jobv, jobq, k, l, tola, tolb,
                fn, is_complex, real_dtype, uses_hidden_lengths):
    """Call ?tgsja.

    LAPACK signature (real):
      DTGSJA(JOBU, JOBV, JOBQ, M, P, N, K, L, A, LDA, B, LDB,
             TOLA, TOLB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ,
             WORK, NCYCLE, INFO)

    WORK is fixed size 2*N (= 2*p_lap).  No workspace query.
    Complex variants add RWORK (size 2*N) after WORK and before NCYCLE.
    gfortran ABI appends three size_t(1) hidden args after INFO.

    Returns (k, l, info).  k and l are passed in (from ggsvp3) but LAPACK
    may update them; we return the post-call values.
    """
    m_lap, p_lap = a_f.shape    # LAPACK M, N
    n_lap        = b_f.shape[0] # LAPACK P

    k_c      = ctypes.c_int(k)
    l_c      = ctypes.c_int(l)
    ncycle_c = ctypes.c_int(0)
    info_c   = ctypes.c_int(0)

    dummy = np.zeros((1, 1), dtype=a_f.dtype, order='F')

    u_ptr   = _ptr(u_f)   if u_f is not None else _ptr(dummy)
    v_ptr   = _ptr(v_f)   if v_f is not None else _ptr(dummy)
    q_ptr   = _ptr(q_f)   if q_f is not None else _ptr(dummy)
    ldu_val = m_lap        if u_f is not None else 1
    ldv_val = n_lap        if v_f is not None else 1
    ldq_val = p_lap        if q_f is not None else 1

    jobu_b = jobu.encode()
    jobv_b = jobv.encode()
    jobq_b = jobq.encode()

    tola_c = ctypes.c_double(tola) if real_dtype == np.float64 else ctypes.c_float(tola)
    tolb_c = ctypes.c_double(tolb) if real_dtype == np.float64 else ctypes.c_float(tolb)

    work = np.zeros(2 * p_lap, dtype=a_f.dtype)

    args = [
        jobu_b, jobv_b, jobq_b,
        _iptr(m_lap), _iptr(n_lap), _iptr(p_lap),
        ctypes.byref(k_c), ctypes.byref(l_c),
        _ptr(a_f),   _iptr(m_lap),
        _ptr(b_f),   _iptr(n_lap),
        ctypes.byref(tola_c), ctypes.byref(tolb_c),
        _ptr(alpha), _ptr(beta),
        u_ptr,       _iptr(ldu_val),
        v_ptr,       _iptr(ldv_val),
        q_ptr,       _iptr(ldq_val),
    ]

    if is_complex and uses_hidden_lengths:
        # Standard gfortran ABI: WORK, RWORK, NCYCLE, INFO
        rwork = np.zeros(2 * p_lap, dtype=real_dtype)
        args += [_ptr(work), _ptr(rwork)]
    else:
        # Real types (any platform) or Accelerate complex: WORK, NCYCLE, INFO (no RWORK)
        args += [_ptr(work)]

    args += [ctypes.byref(ncycle_c), ctypes.byref(info_c)]

    if uses_hidden_lengths:
        one = ctypes.c_size_t(1)
        args += [one, one, one]

    fn(*args)

    return k_c.value, l_c.value, info_c.value


def _ggsvp3_lwork(a_f, b_f):
    """Return a safe work array size for ?ggsvp3 without querying LAPACK.

    The LAPACK minimum is max(3*N, M, P).  We use a generous estimate with a
    typical block size of 64 so that the result is also close to optimal.
    """
    m_lap, p_lap = a_f.shape    # LAPACK M, N
    n_lap        = b_f.shape[0] # LAPACK P
    nb = 64
    return max(3 * p_lap, max(m_lap, n_lap) * nb + p_lap)


def _call_ggsvp3_tgsja(a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
                        jobu, jobv, jobq, tola, tolb, lwork,
                        fn_p3, fn_tgsja, is_complex, real_dtype,
                        uses_hidden_lengths):
    """Two-step GSVD: call ?ggsvp3 then ?tgsja.

    This is equivalent to calling ?ggsvd3, but with explicit tola/tolb.
    Returns (k, l, info).
    """
    # Work size for ggsvp3: use caller-supplied value or compute a safe default.
    # We deliberately avoid the workspace query (LWORK=-1) here because some
    # LAPACK builds (e.g. Accelerate cggsvp3) reject it for complex types.
    if lwork is not None and lwork != -1:
        lwork_use = lwork
    else:
        lwork_use = _ggsvp3_lwork(a_f, b_f)

    # --- Step 1: ggsvp3 ---
    k, l, info = _call_ggsvp3(
        a_f, b_f, u_f, v_f, q_f, iwork,
        jobu, jobv, jobq, tola, tolb, lwork=lwork_use,
        fn=fn_p3, is_complex=is_complex, real_dtype=real_dtype,
        uses_hidden_lengths=uses_hidden_lengths,
    )

    if info != 0:
        return k, l, info

    # --- Step 2: tgsja ---
    k, l, info = _call_tgsja(
        a_f, b_f, alpha, beta, u_f, v_f, q_f,
        jobu, jobv, jobq, k, l, tola, tolb,
        fn=fn_tgsja, is_complex=is_complex, real_dtype=real_dtype,
        uses_hidden_lengths=uses_hidden_lengths,
    )

    return k, l, info


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

def gsvd(a, b, mode='full', compute_u=True, compute_v=True, compute_right=True,
         overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True,
         tola=None, tolb=None):
    """Generalized Singular Value Decomposition.

    Computes the GSVD of the matrix pair (a, b) using the LAPACK routine
    ?ggsvd3 linked via the same LAPACK library as SciPy.

    Parameters
    ----------
    a : (m, p) array_like
    b : (n, p) array_like
    mode : {'full', 'econ', 'separate'}, default 'full'
        'full'     — Full Matlab-style: U (m×m), V (n×n), C (m×q), S (n×q),
                     X (p×q), where q = k+l is the numerical rank of [a; b].
        'econ'     — Economy Matlab-style: U (m×min(m,q)), V (n×min(n,q)),
                     C (min(m,q)×q), S (min(n,q)×q), X (p×q).
        'separate' — Raw LAPACK output (no rank truncation): U, V, D1, D2,
                     R, Q, k, l.
    compute_u : bool, default True
        Compute left singular vectors of a.
    compute_v : bool, default True
        Compute left singular vectors of b.
    compute_right : bool, default True
        Compute the right factor. In 'full'/'econ' mode this is X; in
        'separate' mode this is Q. Setting False sets JOBQ='N' in LAPACK,
        saving an O(p³) accumulation step (significant when p is large).
        R is still returned in 'separate' mode regardless.
    overwrite_a : bool, default False
        Allow overwriting a (avoids a copy if True and a is already
        Fortran-contiguous with the correct dtype).
    overwrite_b : bool, default False
        Allow overwriting b (same as overwrite_a).
    lwork : int or None, default None
        LAPACK work array size.  None (or -1) triggers an optimal query.
    check_finite : bool, default True
        Check that a and b contain only finite values.
    tola : float or None, default None
        Threshold for determining the effective numerical rank of a.
        Values below ``tola`` are treated as zero. If None, uses the
        LAPACK default ``max(m, p) * norm(a, ord=1) * machine_epsilon``.
        Providing tola (or tolb) causes the two lower-level LAPACK routines
        ?ggsvp3 and ?tgsja to be called directly instead of ?ggsvd3.
    tolb : float or None, default None
        Threshold for determining the effective numerical rank of b.
        If None, uses ``max(n, p) * norm(b, ord=1) * machine_epsilon``.

    Returns
    -------
    mode='full' or 'econ':
        C diagonal is in descending order (scipy convention).
        If compute_u and compute_v and compute_right:         U, V, C, S, X
        If compute_u and compute_v and not compute_right:     U, V, C, S
        (other combinations omit U and/or V from the front, and omit X
        from the end when compute_right=False)

    mode='separate':
        If compute_u and compute_v and compute_right:         U, V, D1, D2, R, Q, k, l
        If compute_u and compute_v and not compute_right:     U, V, D1, D2, R, k, l
        (other combinations omit U and/or V from the front, and omit Q
        from the end when compute_right=False)
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
    # Allocate output arrays
    # ------------------------------------------------------------------
    jobu_char = 'U' if compute_u else 'N'
    jobv_char = 'V' if compute_v else 'N'
    jobq_char = 'Q' if compute_right else 'N'

    alpha  = np.zeros(p, dtype=real_dtype)
    beta   = np.zeros(p, dtype=real_dtype)
    iwork  = np.zeros(p, dtype=np.int32)
    q_f    = np.zeros((p, p), dtype=dtype, order='F') if compute_right else None
    u_f    = np.zeros((m, m), dtype=dtype, order='F') if compute_u else None
    v_f    = np.zeros((n, n), dtype=dtype, order='F') if compute_v else None

    # ------------------------------------------------------------------
    # LAPACK call: ?ggsvd3 (default) or ?ggsvp3 + ?tgsja (when tola/tolb set)
    # ------------------------------------------------------------------
    if tola is None and tolb is None:
        # --- default path: single ?ggsvd3 call ---
        fn, uses_hidden_lengths = get_ggsvd3(dtype_char)

        if lwork is None or lwork == -1:
            opt = _call_ggsvd3(
                a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
                jobu_char, jobv_char, jobq_char,
                lwork=None, fn=fn, is_complex=is_complex,
                real_dtype=real_dtype, uses_hidden_lengths=uses_hidden_lengths,
            )
            lwork_use = max(opt, 1)
        else:
            lwork_use = lwork

        k, l, info = _call_ggsvd3(
            a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
            jobu_char, jobv_char, jobq_char,
            lwork=lwork_use, fn=fn, is_complex=is_complex,
            real_dtype=real_dtype, uses_hidden_lengths=uses_hidden_lengths,
        )

        if info < 0:
            raise ValueError(f"Illegal argument #{-info} passed to ?ggsvd3.")
        if info > 0:
            raise np.linalg.LinAlgError(
                f"LAPACK ?ggsvd3 failed to converge (info={info})."
            )

    else:
        # --- tola/tolb path: ?ggsvp3 + ?tgsja ---
        tola_use, tolb_use = _default_tola_tolb(a_f, b_f, real_dtype)
        if tola is not None:
            tola_use = float(tola)
        if tolb is not None:
            tolb_use = float(tolb)

        fn_p3,    uses_hidden_p3    = get_ggsvp3(dtype_char)
        fn_tgsja, uses_hidden_tgsja = get_tgsja(dtype_char)

        k, l, info = _call_ggsvp3_tgsja(
            a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
            jobu_char, jobv_char, jobq_char, tola_use, tolb_use, lwork,
            fn_p3=fn_p3, fn_tgsja=fn_tgsja,
            is_complex=is_complex, real_dtype=real_dtype,
            uses_hidden_lengths=uses_hidden_p3,
        )

        if info < 0:
            raise ValueError(f"Illegal argument #{-info} passed to ?ggsvp3/?tgsja.")
        if info > 0:
            raise np.linalg.LinAlgError(
                f"LAPACK ?tgsja failed to converge (info={info})."
            )

    q_rank = k + l   # effective numerical rank

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    if mode == 'separate':
        return _build_separate(
            a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
            m, n, p, k, l, q_rank,
            compute_u, compute_v, compute_right, real_dtype,
        )
    else:
        return _build_matlab_style(
            a_f, b_f, alpha, beta, u_f, v_f, q_f,
            m, n, p, k, l, q_rank,
            mode, compute_u, compute_v, compute_right,
        )

def gsvdvals(a, b, overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True,
             tola=None, tolb=None):
    """Generalized singular value pairs of the matrix pair (a, b).

    Computes the generalized cosines ``c`` and sines ``s`` that satisfy
    ``c[i]**2 + s[i]**2 = 1`` for the finite generalized singular values.
    The generalized singular values themselves are the ratios ``c[i] / s[i]``.

    Equivalent to calling ``gsvd(a, b, mode='econ', compute_u=False,
    compute_v=False, compute_right=False)`` and extracting the diagonals of
    the returned ``C`` and ``S`` matrices, but with a cleaner interface.

    Parameters
    ----------
    a : (m, p) array_like
        First input matrix.
    b : (n, p) array_like
        Second input matrix.
    overwrite_a : bool, default False
        Allow overwriting a (avoids a copy if True and a is already
        Fortran-contiguous with the correct dtype).
    overwrite_b : bool, default False
        Allow overwriting b (same as overwrite_a).
    lwork : int or None, default None
        LAPACK work array size. None (or -1) triggers an optimal query.
    check_finite : bool, default True
        Check that a and b contain only finite values.
    tola : float or None, default None
        Rank threshold for a; passed directly to gsvd().
    tolb : float or None, default None
        Rank threshold for b; passed directly to gsvd().

    Returns
    -------
    c : ndarray, shape (q,)
        Generalized cosines in non-increasing order, where ``q = k + l`` is
        the numerical rank of ``[a; b]``. An entry equal to 1 indicates an
        infinite generalized singular value; an entry equal to 0 indicates a
        zero generalized singular value.
    s : ndarray, shape (q,)
        Generalized sines in non-decreasing order. An entry equal to 0
        indicates an infinite generalized singular value; an entry equal to 1
        indicates a zero generalized singular value.
    """
    # get singular value matrices only
    C, S = gsvd(a, b, 'econ', False, False, False, overwrite_a,
                overwrite_b, lwork, check_finite, tola, tolb)

    # one singular value pair per column of C, S
    c, s = np.max(C, axis=0), np.max(S, axis=0)

    return c, s

# ---------------------------------------------------------------------------
# Post-processing: separate mode
# ---------------------------------------------------------------------------

def _build_separate(a_f, b_f, alpha, beta, u_f, v_f, q_f, iwork,
                    m, n, p, k, l, q_rank,
                    compute_u, compute_v, compute_right, real_dtype):
    R = _extract_R(a_f, b_f, m, n, p, k, l)
    D1, D2 = _build_C_S(alpha, beta, m, n, k, l)

    # Convert to C-order for return
    R  = np.ascontiguousarray(R)
    D1 = np.ascontiguousarray(D1)
    D2 = np.ascontiguousarray(D2)

    result = []
    if compute_u:
        result.append(np.ascontiguousarray(u_f))
    if compute_v:
        result.append(np.ascontiguousarray(v_f))
    result += [D1, D2, R]
    if compute_right:
        result.append(np.ascontiguousarray(q_f))
    result += [k, l]
    return tuple(result)


# ---------------------------------------------------------------------------
# Post-processing: full / econ modes
# ---------------------------------------------------------------------------

def _sort_gsvd_outputs(C, S, X, U, V, k, m, n, compute_u, compute_v):
    """Sort the finite GSV block so the diagonal of C is in descending order.

    Only columns k..k+finite_len-1 are permuted (the finite cosine block).
    The k infinite-GSV columns (c=1) stay first; the zero-c columns stay last.
    Corresponding rows/columns of U and V are permuted to preserve A=UCX^H
    and B=VSX^H.
    """
    q = C.shape[1]
    finite_len = min(m, q) - k
    if finite_len <= 1:
        return C, S, X, U, V

    c_finite = np.array([C[k + i, k + i] for i in range(finite_len)])
    s_finite = np.array([S[i, k + i] for i in range(finite_len)])
    # Primary sort: c descending. Secondary sort: s ascending (breaks ties in c,
    # e.g. multiple alpha==1.0 entries whose beta values may be unsorted).
    perm = np.lexsort((s_finite, -c_finite))
    if np.all(perm == np.arange(finite_len)):
        return C, S, X, U, V

    # Permute finite block columns of X (if X was computed)
    if X is not None:
        X = np.array(X)
        X[:, k:k + finite_len] = X[:, k:k + finite_len][:, perm]

    # Update C diagonal in the finite block
    C = np.array(C)
    c_new = c_finite[perm]
    for i in range(finite_len):
        C[k + i, k + i] = c_new[i]

    # Update S finite block (rows 0..finite_len-1, cols k..k+finite_len-1)
    # S[i, k+i] = beta[k+i] for i in 0..finite_len-1
    S = np.array(S)
    S[:finite_len, k:k + finite_len] = 0
    s_new = s_finite[perm]
    for i in range(finite_len):
        S[i, k + i] = s_new[i]

    # Permute U columns k..k+finite_len-1
    if compute_u and U is not None:
        ue = min(k + finite_len, U.shape[1])
        if ue > k:
            blen = ue - k
            U = np.array(U)
            U[:, k:ue] = U[:, k:ue][:, perm[:blen]]

    # Permute V columns 0..finite_len-1 (correspond to S rows 0..finite_len-1)
    if compute_v and V is not None:
        ve = min(finite_len, V.shape[1])
        if ve > 0:
            V = np.array(V)
            V[:, :ve] = V[:, :ve][:, perm[:ve]]

    return C, S, X, U, V


def _build_matlab_style(a_f, b_f, alpha, beta, u_f, v_f, q_f,
                        m, n, p, k, l, q_rank,
                        mode, compute_u, compute_v, compute_right):
    # Build C and S (real-valued diagonal matrices)
    C_full, S_full = _build_C_S(alpha, beta, m, n, k, l)

    # Full mode: U is m×m, V is n×n, C is m×q, S is n×q
    # Econ mode: truncate U to m×r, V to n×r, C to r×q, S to r×q
    #   where r = min(m, q_rank) for U/C  and  min(n, q_rank) for V/S
    if mode == 'full':
        C = np.ascontiguousarray(C_full)
        S = np.ascontiguousarray(S_full)
        U_out = np.ascontiguousarray(u_f) if compute_u else None
        V_out = np.ascontiguousarray(v_f) if compute_v else None
    else:  # 'econ'
        ru = min(m, q_rank)
        rv = min(n, q_rank)
        C  = np.ascontiguousarray(C_full[:ru, :])
        S  = np.ascontiguousarray(S_full[:rv, :])
        U_out = np.ascontiguousarray(u_f[:, :ru]) if compute_u else None
        V_out = np.ascontiguousarray(v_f[:, :rv]) if compute_v else None

    # Build X = Q2 @ conj(R).T only when compute_right=True
    if compute_right:
        R  = _extract_R(a_f, b_f, m, n, p, k, l)
        Q2 = np.asarray(q_f)[:, p - q_rank:]    # p×q_rank
        X  = np.ascontiguousarray(Q2 @ np.conj(R).T)
    else:
        X = None

    # Sort finite GSV block so diagonal of C is in descending order
    C, S, X, U_out, V_out = _sort_gsvd_outputs(
        C, S, X, U_out, V_out, k, m, n, compute_u, compute_v
    )

    result = []
    if compute_u:
        result.append(U_out)
    if compute_v:
        result.append(V_out)
    result += [C, S]
    if compute_right:
        result.append(X)
    return tuple(result)
