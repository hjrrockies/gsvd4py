"""
LAPACK library discovery for gsvd4py.

Tries, in order:
  1. Apple Accelerate (macOS) — symbols named ?ggsvd3$NEWLAPACK
  2. scipy_openblas32           — symbols named scipy_?ggsvd3_
  3. scipy_openblas64           — symbols named scipy_?ggsvd3_
  4. CDLL(None)                 — all loaded symbols (works on Linux)
  5. ctypes.util.find_library   — system LAPACK / OpenBLAS

Calling conventions differ:
  - Accelerate:        pure C interface, no hidden Fortran char-length args
  - gfortran LAPACK:   three hidden size_t args (len_jobu, len_jobv, len_jobq)
                       appended after `info`
"""

import ctypes
import ctypes.util
import glob
import os
import sys

# Module-level cache
_lib = None
_lib_type = None   # 'accelerate' | 'scipy_openblas' | 'system'


def _load_lib():
    global _lib, _lib_type

    if _lib is not None:
        return

    # --- Strategy 1: Apple Accelerate (macOS) ---
    if sys.platform == 'darwin':
        try:
            lib = ctypes.CDLL(
                '/System/Library/Frameworks/Accelerate.framework/Accelerate'
            )
            lib['dggsvd3$NEWLAPACK']   # raises KeyError if absent
            _lib = lib
            _lib_type = 'accelerate'
            return
        except (OSError, KeyError):
            pass

    # --- Strategy 2 & 3: scipy_openblas32 / scipy_openblas64 ---
    for _pkg in ('scipy_openblas32', 'scipy_openblas64'):
        try:
            pkg = __import__(_pkg)
            lib_dir = pkg.get_lib_dir()
            pattern = '*.dylib' if sys.platform == 'darwin' else '*.so*'
            for dylib in glob.glob(os.path.join(lib_dir, pattern)):
                try:
                    lib = ctypes.CDLL(dylib)
                    getattr(lib, f'scipy_dggsvd3_')
                    _lib = lib
                    _lib_type = 'scipy_openblas'
                    return
                except (OSError, AttributeError):
                    pass
        except ImportError:
            pass

    # --- Strategy 4: CDLL(None) — all loaded symbols (Linux) ---
    lib = ctypes.CDLL(None)
    try:
        getattr(lib, 'dggsvd3_')
        _lib = lib
        _lib_type = 'system'
        return
    except AttributeError:
        pass

    # --- Strategy 5: find_library ---
    for name in ('lapack', 'openblas', 'flexiblas'):
        path = ctypes.util.find_library(name)
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
            getattr(lib, 'dggsvd3_')
            _lib = lib
            _lib_type = 'system'
            return
        except (OSError, AttributeError):
            pass

    raise ImportError(
        "gsvd4py: Could not find a LAPACK library providing dggsvd3. "
        "Ensure scipy is installed (pip install scipy), or install "
        "scipy-openblas32 (pip install scipy-openblas32)."
    )


def _get_lapack_fn(base_name, dtype_char):
    """Return (fn, uses_hidden_lengths) for a LAPACK routine.

    Parameters
    ----------
    base_name : str
        Routine name without the leading dtype char, e.g. 'ggsvd3'.
    dtype_char : str
        One of 'd', 's', 'z', 'c'.
    """
    _load_lib()

    if _lib_type == 'accelerate':
        sym = f'{dtype_char}{base_name}$NEWLAPACK'
        fn = _lib[sym]
        uses_hidden_lengths = False
    elif _lib_type == 'scipy_openblas':
        sym = f'scipy_{dtype_char}{base_name}_'
        fn = getattr(_lib, sym)
        uses_hidden_lengths = True
    else:   # 'system'
        sym = f'{dtype_char}{base_name}_'
        fn = getattr(_lib, sym)
        uses_hidden_lengths = True

    fn.restype = None
    return fn, uses_hidden_lengths


def get_ggsvd3(dtype_char):
    """Return the ctypes function handle for ?ggsvd3.

    Parameters
    ----------
    dtype_char : str
        One of 'd', 's', 'z', 'c'.

    Returns
    -------
    fn : ctypes function object (restype already set to None)
    uses_hidden_lengths : bool
        True when the function uses the gfortran hidden char-length ABI.
    """
    return _get_lapack_fn('ggsvd3', dtype_char)


def get_ggsvp3(dtype_char):
    """Return the ctypes function handle for ?ggsvp3.

    Parameters
    ----------
    dtype_char : str
        One of 'd', 's', 'z', 'c'.

    Returns
    -------
    fn : ctypes function object (restype already set to None)
    uses_hidden_lengths : bool
        True when the function uses the gfortran hidden char-length ABI.
    """
    return _get_lapack_fn('ggsvp3', dtype_char)


def get_tgsja(dtype_char):
    """Return the ctypes function handle for ?tgsja.

    Parameters
    ----------
    dtype_char : str
        One of 'd', 's', 'z', 'c'.

    Returns
    -------
    fn : ctypes function object (restype already set to None)
    uses_hidden_lengths : bool
        True when the function uses the gfortran hidden char-length ABI.
    """
    return _get_lapack_fn('tgsja', dtype_char)
