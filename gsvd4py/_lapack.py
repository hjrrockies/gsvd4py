"""
LAPACK library discovery for gsvd4py.

Tries, in order:
  1. Apple Accelerate (macOS) — symbols named ?ggsvd3$NEWLAPACK
  2. SciPy's own bundled OpenBLAS (scipy.libs/ or scipy/.dylibs/)
  3. scipy_openblas32 / scipy_openblas64 packages
  4. CDLL(None)                 — all loaded symbols (POSIX only)
  5. ctypes.util.find_library   — system LAPACK / OpenBLAS

Strategy 2 is what makes the "same LAPACK as SciPy" promise hold on Linux
and Windows: the wheels vendor libscipy_openblas next to the scipy package,
exporting scipy_?ggsvd3_ just like the standalone scipy_openblas32 package.

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


def _shared_lib_pattern():
    """Glob pattern matching shared libraries on this platform."""
    if sys.platform == 'darwin':
        return '*.dylib'
    if sys.platform == 'win32':
        return '*.dll'
    return '*.so*'


def _try_load(path):
    """Load `path` and return (lib, lib_type), or None if it has no ?ggsvd3.

    Probes both the scipy_-prefixed and the plain Fortran symbol, so the same
    helper handles SciPy's vendored OpenBLAS and an unprefixed system LAPACK.
    """
    try:
        if sys.platform == 'win32':
            # Let the loader find any DLLs the library depends on.
            with os.add_dll_directory(os.path.dirname(path)):
                lib = ctypes.CDLL(path, winmode=0)
        else:
            lib = ctypes.CDLL(path)
    except (OSError, TypeError):
        return None

    for sym, lib_type in (('scipy_dggsvd3_', 'scipy_openblas'),
                          ('dggsvd3_', 'system')):
        try:
            getattr(lib, sym)
            return lib, lib_type
        except AttributeError:
            pass
    return None


def _scipy_bundled_lib_dirs():
    """Directories where a SciPy wheel vendors its OpenBLAS build."""
    try:
        import scipy
    except ImportError:
        return []

    pkg_dir  = os.path.dirname(os.path.abspath(scipy.__file__))
    site_dir = os.path.dirname(pkg_dir)
    return [
        os.path.join(site_dir, 'scipy.libs'),   # auditwheel / delvewheel
        os.path.join(pkg_dir, '.dylibs'),       # delocate (macOS wheels)
    ]


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

    # --- Strategy 2: the OpenBLAS bundled inside the SciPy wheel ---
    pattern = _shared_lib_pattern()
    for lib_dir in _scipy_bundled_lib_dirs():
        # Prefer the OpenBLAS itself over the gfortran/quadmath libs beside it.
        paths = sorted(glob.glob(os.path.join(lib_dir, pattern)),
                       key=lambda p: 'openblas' not in os.path.basename(p).lower())
        for path in paths:
            found = _try_load(path)
            if found is not None:
                _lib, _lib_type = found
                return

    # --- Strategy 3: scipy_openblas32 / scipy_openblas64 packages ---
    for _pkg in ('scipy_openblas32', 'scipy_openblas64'):
        try:
            pkg = __import__(_pkg)
        except ImportError:
            continue
        for path in glob.glob(os.path.join(pkg.get_lib_dir(), pattern)):
            found = _try_load(path)
            if found is not None:
                _lib, _lib_type = found
                return

    # --- Strategy 4: CDLL(None) — all loaded symbols (POSIX only) ---
    if sys.platform != 'win32':
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
        found = _try_load(path)
        if found is not None:
            _lib, _lib_type = found
            return

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
