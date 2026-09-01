# gsvd4py

[![PyPI version](https://img.shields.io/pypi/v/gsvd4py.svg)](https://pypi.org/project/gsvd4py/)
[![tests](https://github.com/hjrrockies/gsvd4py/actions/workflows/tests.yml/badge.svg)](https://github.com/hjrrockies/gsvd4py/actions/workflows/tests.yml)

A lightweight Python wrapper for the LAPACK `?ggsvd3` routines, providing the Generalized Singular Value Decomposition (GSVD) in a style similar to `scipy.linalg`. It links to the same LAPACK library that SciPy uses on your machine — no separate LAPACK installation required.

## Installation

```bash
pip install gsvd4py
```

Requires Python >= 3.9, SciPy >= 1.7.3 and NumPy >= 1.19.5. Tested on Linux,
macOS and Windows across Python 3.9-3.13; the oldest supported versions are
pinned and exercised by CI, so the requirements above are what is actually
verified rather than a guess.

SciPy is needed only to locate a LAPACK library — no SciPy API is called.

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

### Skipping X (or Q in `mode='separate'`)
To retrieve the full diagonal matrices `C` and `S` alongside singular vectors,
set `compute_right=False` on `gsvd`. This skips the accumulation of `X`
and can give a significant speedup when `p` is large:

```python
U, V, C, S = gsvd(A, B, compute_right=False)

# In separate mode, R is still returned; only Q is omitted:
U, V, D1, D2, R, k, l = gsvd(A, B, mode='separate', compute_right=False)
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

Rank determination uses LAPACK's own tolerances,
`max(m, p) * norm(a, ord=1) * eps` and `max(n, p) * norm(b, ord=1) * eps`.

### `gsvdvals`

```python
gsvdvals(a, b, overwrite_a=False, overwrite_b=False, lwork=None,
         check_finite=True)
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

`gsvd4py` calls LAPACK through `ctypes`, so no compilation is required and no
LAPACK is bundled — whatever the host already provides is used. The library is
discovered at import in this order:

1. **Apple Accelerate** (macOS) — via `$NEWLAPACK` symbols
2. **SciPy's own bundled OpenBLAS** — `scipy.libs/` beside the installed
   `scipy` package (`scipy/.dylibs/` for macOS wheels). This is what makes
   "the same LAPACK SciPy uses" true on Linux and Windows.
3. **`scipy_openblas32` / `scipy_openblas64`** — the standalone packages
4. **Already-loaded symbols** — `CDLL(None)`, POSIX only
5. **System LAPACK** — `liblapack`, `libopenblas` or `libflexiblas` found via
   `ctypes.util.find_library`

Both the `scipy_`-prefixed and plain Fortran symbol spellings are accepted, and
libraries in a bundle directory are loaded together so that co-located
dependencies (`libgfortran`, `libquadmath`) can satisfy each other. If every
candidate fails, the resulting `ImportError` lists each path tried and why it
was rejected.

You can check what was selected:

```python
import gsvd4py._lapack as lapack
lapack._load_lib()
print(lapack._lib_type, lapack._lib)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
