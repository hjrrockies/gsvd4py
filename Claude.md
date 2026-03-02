# Project Description
`gsvd4py` is intended to be a lightweight python module which wraps the LAPACK routines for the Generalized Singular Value Decomposition and makes them available in a style similar to scipy.linalg. Its sole purpose is to accomplish this in a user-friendly way, so that someone who already has SciPy installed can run `pip install gsvd4py` and reliably connect to the same LAPACK libraries that SciPy is using on that machine.

# Key Mathematical Points
- The relevant LAPACK routines are of the form ?ggsvd3, which are dggsvd3, zggsvd3, sggsvd3, and cggsvd3. The documentation for ggsvd3 is found at this url: https://www.netlib.org/lapack/explore-html/d1/d27/group__ggsvd3.html
- The GSVD applies to a pair of matrices A & B, where A is m-by-p and B is n-by-p
- The GSVD has several forms:
    - The LAPACK routines compute the "full" GSVD (see LAPACK documentation)
    - The "full Matlab-style" GSVD is $A = U C X^H, B = V S X^H$, where the matrices have the following properties (note that q = K+L, the numerical rank of the stacked matrices)
        - U is m-by-m unitary
        - V is n-by-n unitary
        - C is m-by-q diagonal, with a non-increasing diagonal
        - S is n-by-q diagonal, with a non-decreasing diagonal
        - X is p-by-q
    - The "economy Matlab-style" GSVD is also $A = U C X^H, B = V S X^H$, but U and V are truncated to at most p columns, and C and S are truncated to at most p rows

# Module syntax
- gsvd4py should export a single function `gsvd` which has the following syntax:
```python
def gsvd(a, b, mode='full', compute_u=True, compute_v=True, compute_right=True,
         overwrite_a=False, overwrite_b=False, lwork=None, check_finite=True):
         # GSVD code
```
- a and b are the matrices
- if `mode='full'`, it returns the full Matlab-style GSVD (with numerical rank truncation)
- if `mode='econ'`, it returns the economy Matlab-style GSVD (with numerical rank truncation)
- if `mode='separate'`, it returns the LAPACK GSVD (without numerical rank truncation)
- `compute_u` and `compute_v` control whether or not to compute the left singular vectors
- `compute_right` controls whether or not to return the right singular vector matrix X in the mode='full' and mode='econ' cases, and whether or not to return Q in the mode=`separate` case
- `overwrite_a` and `overwrite_b` allow for a slight performance speedup when set to `True`
- `lwork` controls the work array size, and is computed optimally if `lwork=None` or `lwork=-1`
- `check_finite=False` allows the user to gain a slight performance boost by skipping the checks to see if `a` and `b` contain only finite values
- see the following examples for calling `gsvd`:
```python
U, V, C, S, X = gsvd(A, B)
U, V, C, S, X = gsvd(A, B, mode='econ')
C, S, X = gsvd(A, B, compute_u=False, compute_v=False)
U, V, C, S = gsvd(A, B, compute_right=False)
U, V, D1, D2, R, Q, k, l = gsvd(A, B, mode='separate')
U, V, D1, D2, R, k, l = gsvd(A, B, mode='separate', compute_right=False)
```

# Build requirements
- The module should mirror SciPy/NumPy behavior (as of SciPy 1.17) in linking to a LAPACK installation. 
- In particular, it should follow SciPy and link to `scipy-openblas` if SciPy does.
- If SciPy links to a different LAPACK installation (such as Apple Accelerate), gsvd4py should do the same.

# Testing
- The module should include unit tests which validate the build process and correct computation of the GSVD

# Virtual environment (for development)
- I am using uv to manage the virtual environment, which can be accessed using `source .venv/bin/activate`
- Development testing should be performed within this virtual environment

# Dependencies
- The module should work with SciPy version 1.13 or higher, which corresponds to NumPy 2.0 or higher. See `pyproject.toml`.