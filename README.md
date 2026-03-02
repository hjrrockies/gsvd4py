# gsvd4py

[![PyPI version](https://img.shields.io/pypi/v/gsvd4py.svg)](https://pypi.org/project/gsvd4py/)

A lightweight Python wrapper for the LAPACK `?ggsvd3` routines, providing the Generalized Singular Value Decomposition (GSVD) in a style similar to `scipy.linalg`. It links to the same LAPACK library that SciPy uses on your machine — no separate LAPACK installation required.

## Installation

```bash
pip install gsvd4py
```

Requires SciPy >= 1.13 and NumPy >= 2.0.

## Background

The GSVD decomposes a pair of matrices `A` (m×p) and `B` (n×p) as:

```
A = U @ C @ X.conj().T
B = V @ S @ X.conj().T
```

where:
- `U` (m×m) and `V` (n×n) are unitary
- `C` (m×q) and `S` (n×q) are real diagonal, with the diagonal of `C` in descending order and `C.T @ C + S.T @ S = I`
- `X` (p×q) is nonsingular
- `q = k + l` is the numerical rank of the stacked matrix `[A; B]`

The generalized singular values are the ratios `C[i,i] / S[i,i]`.

## Usage

```python
import numpy as np
from gsvd4py import gsvd

A = np.random.randn(5, 6)
B = np.random.randn(4, 6)
```

### Full GSVD (default)

```python
U, V, C, S, X = gsvd(A, B)
# U: (5,5), V: (4,4), C: (5,q), S: (4,q), X: (6,q)
# diagonal of C is in descending order
```

### Economy GSVD

Truncates `U` and `V` to at most `q` columns:

```python
U, V, C, S, X = gsvd(A, B, mode='econ')
```

### Raw LAPACK output

Returns the LAPACK decomposition `A = U @ D1 @ [0, R] @ Q.T` directly:

```python
U, V, D1, D2, R, Q, k, l = gsvd(A, B, mode='separate')
```

### Skipping U and/or V

```python
C, S, X = gsvd(A, B, compute_u=False, compute_v=False)
U, C, S, X = gsvd(A, B, compute_v=False)
V, C, S, X = gsvd(A, B, compute_u=False)
```

### Generalized singular values only

Use `gsvdvals` to get just the generalized cosine/sine pairs `(c, s)` without
computing any singular vectors or the right factor `X`:

```python
from gsvd4py import gsvdvals

c, s = gsvdvals(A, B)
# c[i]**2 + s[i]**2 == 1; generalized singular values are c[i] / s[i]
# c is non-increasing (equivalently, s is non-decreasing)
```

To retrieve the full diagonal matrices `C` and `S` alongside singular vectors,
set `compute_right=False` on `gsvd`. This skips the O(p³) accumulation of `X`
and can give a significant speedup when `p` is large:

```python
U, V, C, S = gsvd(A, B, compute_right=False)

# In separate mode, R is still returned; only Q is omitted:
U, V, D1, D2, R, k, l = gsvd(A, B, mode='separate', compute_right=False)
```

## API Reference

### `gsvd`

```python
gsvd(a, b, mode='full', compute_u=True, compute_v=True, compute_right=True,
     overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True)
```

| Parameter | Description |
|-----------|-------------|
| `a` | (m, p) array |
| `b` | (n, p) array |
| `mode` | `'full'` (default), `'econ'`, or `'separate'` |
| `compute_u` | Compute left singular vectors of `a` (default `True`) |
| `compute_v` | Compute left singular vectors of `b` (default `True`) |
| `compute_right` | Compute `X` (or `Q` in `separate` mode); set `False` to skip the O(p³) accumulation (default `True`) |
| `overwrite_a` | Allow overwriting `a` to avoid a copy (default `False`) |
| `overwrite_b` | Allow overwriting `b` to avoid a copy (default `False`) |
| `lwork` | Work array size; `None` triggers an optimal workspace query |
| `check_finite` | Check inputs for non-finite values (default `True`) |

### `gsvdvals`

```python
gsvdvals(a, b, overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True)
```

Returns `(c, s)` — 1D real arrays of length `q = k + l` (the numerical rank of
`[a; b]`) containing the generalized cosines and sines in non-increasing /
non-decreasing order respectively. Parameters have the same meaning as for
`gsvd`.

| Return value | Description |
|---|---|
| `c` | Generalized cosines, shape (q,), non-increasing. `c[i] == 1` ↔ infinite GSV; `c[i] == 0` ↔ zero GSV. |
| `s` | Generalized sines, shape (q,), non-decreasing. `s[i] == 0` ↔ infinite GSV; `s[i] == 1` ↔ zero GSV. |

Supported dtypes: `float32`, `float64`, `complex64`, `complex128`. Integer inputs are upcast to `float64`.

## LAPACK backend

`gsvd4py` discovers the LAPACK library at runtime in the following order:

1. **Apple Accelerate** (macOS) — via `$NEWLAPACK` symbols
2. **scipy-openblas** — the OpenBLAS bundle shipped with SciPy
3. **System LAPACK** — `liblapack` found via `ctypes.util.find_library`

No compilation is required.

## License

MIT
