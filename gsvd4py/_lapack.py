"""
LAPACK library discovery for gsvd4py.

Discovery runs once, on the first call that needs LAPACK (not at import).
Each strategy below is a named entry in `_STRATEGIES`; by default they are
tried in this order:
  1. accelerate        Apple Accelerate (macOS) — symbols ?ggsvd3$NEWLAPACK
  2. scipy_bundled     SciPy's own bundled OpenBLAS (scipy.libs/, scipy/.dylibs/)
  3. scipy_openblas32  the standalone scipy_openblas32 package
  4. process_symbols   CDLL(None) — symbols already loaded (POSIX only)
  5. find_library      ctypes.util.find_library — system LAPACK / OpenBLAS

Strategy 2 is what makes the "same LAPACK as SciPy" promise hold on Linux
and Windows: the wheels vendor libscipy_openblas next to the scipy package,
exporting scipy_?ggsvd3_ just like the standalone scipy_openblas32 package.

scipy_openblas64 is deliberately never used: it is an ILP64 build, and
gsvd4py passes 32-bit integers.

The environment variable GSVD4PY_LAPACK pins the provider. It is read once,
at first use, and accepts:
  - accelerate       only strategy 1
  - scipy_openblas   only strategies 2 and 3
  - system           only strategies 4 and 5
  - an absolute path to a shared library exporting ?ggsvd3
An explicit provider that cannot be loaded raises ImportError; it never
falls back to another one.

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

_ACCELERATE_PATH = '/System/Library/Frameworks/Accelerate.framework/Accelerate'
_ENV_VAR = 'GSVD4PY_LAPACK'

# Module-level cache
_lib = None
_lib_type = None     # 'accelerate' | 'scipy_openblas' | 'system'
_lib_path = None     # path of the loaded library; None for process symbols
_lib_source = None   # 'override' | 'default'


def _shared_lib_pattern():
    """Glob pattern matching shared libraries on this platform."""
    if sys.platform == 'darwin':
        return '*.dylib'
    if sys.platform == 'win32':
        return '*.dll'
    return '*.so*'


def _dlopen(path):
    """Open one shared library, handling the Windows DLL search path.

    RTLD_GLOBAL (a no-op on Windows) lets a library loaded here satisfy the
    dependencies of one loaded afterwards.
    """
    if sys.platform == 'win32':
        # Let the loader find any DLLs the library depends on.
        with os.add_dll_directory(os.path.dirname(path)):
            return ctypes.CDLL(path, winmode=0)
    return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _has_symbol(lib, sym):
    """True if `lib` exports `sym` (which may contain '$')."""
    try:
        lib[sym]
        return True
    except AttributeError:
        return False


def _identify(lib, accelerate=False):
    """Return the lib_type whose ?ggsvd3 spelling `lib` exports, or None.

    Probes both the scipy_-prefixed and the plain Fortran symbol, so the same
    helper handles SciPy's vendored OpenBLAS and an unprefixed system LAPACK.
    The Accelerate spelling is only checked when asked for.
    """
    candidates = [('scipy_dggsvd3_', 'scipy_openblas'),
                  ('dggsvd3_', 'system')]
    if accelerate:
        candidates.insert(0, ('dggsvd3$NEWLAPACK', 'accelerate'))
    for sym, lib_type in candidates:
        if _has_symbol(lib, sym):
            return lib_type
    return None


def _try_load(path, attempts, accelerate=False):
    """Load `path`; return (lib, lib_type, path), or None if it has no ?ggsvd3.

    Records why a candidate was rejected in `attempts`, for the error message
    raised when nothing works.
    """
    try:
        lib = _dlopen(path)
    except (OSError, TypeError) as exc:
        attempts.append(f"{path}: could not load ({exc})")
        return None

    lib_type = _identify(lib, accelerate=accelerate)
    if lib_type is not None:
        return lib, lib_type, path

    attempts.append(f"{path}: loaded, but exports no dggsvd3")
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


def _load_from_dir(paths, attempts):
    """Load the LAPACK provider out of a directory of co-located libraries.

    Older SciPy wheels (<= 1.13) bundle libopenblasp alongside the libgfortran
    and libquadmath it depends on, without a usable RPATH -- so loading the
    OpenBLAS directly fails until its dependencies are already in the process.
    Load everything with RTLD_GLOBAL, retrying until no further progress is
    made, so each pass can satisfy the next one's dependencies; then probe.
    """
    loaded, remaining = [], list(paths)

    while remaining:
        progress, still = False, []
        for path in remaining:
            try:
                loaded.append((path, _dlopen(path)))
                progress = True
            except (OSError, TypeError) as exc:
                still.append((path, exc))
        if not progress:
            attempts.extend(f"{path}: could not load ({exc})"
                            for path, exc in still)
            break
        remaining = [path for path, _ in still]

    for path, lib in loaded:
        lib_type = _identify(lib)
        if lib_type is not None:
            return lib, lib_type, path
        attempts.append(f"{path}: loaded, but exports no dggsvd3_")
    return None


# ---------------------------------------------------------------------------
# Strategies: each takes `attempts` and returns (lib, lib_type, path) or None
# ---------------------------------------------------------------------------

def _strategy_accelerate(attempts):
    """Apple Accelerate, via its $NEWLAPACK symbols (macOS 13.3+)."""
    if sys.platform != 'darwin':
        attempts.append("Accelerate.framework: not on macOS")
        return None
    try:
        lib = ctypes.CDLL(_ACCELERATE_PATH)
    except OSError as exc:
        attempts.append(f"Accelerate.framework: {exc}")
        return None
    if not _has_symbol(lib, 'dggsvd3$NEWLAPACK'):
        attempts.append("Accelerate.framework: no dggsvd3$NEWLAPACK "
                        "(requires macOS 13.3 or newer)")
        return None
    return lib, 'accelerate', _ACCELERATE_PATH


def _strategy_scipy_bundled(attempts):
    """The OpenBLAS bundled inside the SciPy wheel."""
    pattern = _shared_lib_pattern()
    bundled_dirs = _scipy_bundled_lib_dirs()
    if not bundled_dirs:
        attempts.append("scipy is not importable, so its bundled LAPACK "
                        "could not be searched")
    for lib_dir in bundled_dirs:
        if not os.path.isdir(lib_dir):
            attempts.append(f"{lib_dir}: no such directory")
            continue
        paths = glob.glob(os.path.join(lib_dir, pattern))
        if not paths:
            attempts.append(f"{lib_dir}: no {pattern} files")
            continue
        found = _load_from_dir(paths, attempts)
        if found is not None:
            return found
    return None


def _strategy_scipy_openblas32(attempts):
    """The standalone scipy_openblas32 package.

    scipy_openblas64 is not probed: it is ILP64, and gsvd4py passes c_int.
    """
    try:
        import scipy_openblas32 as pkg
    except ImportError:
        attempts.append("scipy_openblas32: not installed")
        return None
    pattern = os.path.join(pkg.get_lib_dir(), _shared_lib_pattern())
    for path in glob.glob(pattern):
        found = _try_load(path, attempts)
        if found is not None:
            return found
    return None


def _strategy_process_symbols(attempts):
    """CDLL(None): LAPACK symbols already loaded into the process (POSIX)."""
    if sys.platform == 'win32':
        return None
    lib = ctypes.CDLL(None)
    if _has_symbol(lib, 'dggsvd3_'):
        return lib, 'system', None
    attempts.append("process symbols: no dggsvd3_ already loaded")
    return None


def _strategy_find_library(attempts):
    """A system LAPACK located by ctypes.util.find_library."""
    for name in ('lapack', 'openblas', 'flexiblas'):
        path = ctypes.util.find_library(name)
        if not path:
            attempts.append(f"find_library({name!r}): not found")
            continue
        found = _try_load(path, attempts)
        if found is not None:
            return found
    return None


_STRATEGIES = {
    'accelerate':       _strategy_accelerate,
    'scipy_bundled':    _strategy_scipy_bundled,
    'scipy_openblas32': _strategy_scipy_openblas32,
    'process_symbols':  _strategy_process_symbols,
    'find_library':     _strategy_find_library,
}

_DEFAULT_ORDER = ('accelerate', 'scipy_bundled', 'scipy_openblas32',
                  'process_symbols', 'find_library')

# GSVD4PY_LAPACK provider name -> the only strategies it may use
_OVERRIDE_GROUPS = {
    'accelerate':     ('accelerate',),
    'scipy_openblas': ('scipy_bundled', 'scipy_openblas32'),
    'system':         ('process_symbols', 'find_library'),
}


def _probe_order(preferred=None):
    """Strategy names to try, in order, given SciPy's detected provider."""
    return list(_DEFAULT_ORDER)


def _read_override():
    """Return the GSVD4PY_LAPACK value, stripped, or None if unset/empty."""
    value = os.environ.get(_ENV_VAR, '').strip()
    return value or None


def _raise_not_found(attempts, headline):
    detail = "\n  ".join(attempts) if attempts else "no candidates found"
    raise ImportError(f"gsvd4py: {headline}\nSearched:\n  {detail}")


def _load_lib():
    global _lib, _lib_type, _lib_path, _lib_source

    if _lib is not None:
        return

    attempts = []   # notes on each rejected candidate, for the error message
    override = _read_override()

    if override is None:
        names, source = _probe_order(), 'default'
    elif override.lower() in _OVERRIDE_GROUPS:
        names, source = _OVERRIDE_GROUPS[override.lower()], 'override'
    elif os.path.isabs(override):
        found = _try_load(override, attempts, accelerate=True)
        if found is None:
            _raise_not_found(
                attempts,
                f"{_ENV_VAR}={override!r} does not provide dggsvd3.")
        _lib, _lib_type, _lib_path = found
        _lib_source = 'override'
        return
    else:
        choices = ", ".join(sorted(_OVERRIDE_GROUPS))
        raise ImportError(
            f"gsvd4py: invalid {_ENV_VAR}={override!r}; expected one of "
            f"{choices}, or an absolute path to a shared library.")

    for name in names:
        found = _STRATEGIES[name](attempts)
        if found is not None:
            _lib, _lib_type, _lib_path = found
            _lib_source = source
            return

    if source == 'override':
        _raise_not_found(
            attempts,
            f"{_ENV_VAR}={override!r} was requested, but no such LAPACK "
            "providing dggsvd3 could be loaded.")
    _raise_not_found(
        attempts,
        "Could not find a LAPACK library providing dggsvd3. Ensure scipy "
        "is installed (pip install scipy), or install scipy-openblas32 "
        "(pip install scipy-openblas32).")


def lapack_info():
    """Describe the LAPACK library gsvd4py uses, loading it if necessary.

    Include this in bug reports. Discovery can be pinned with the
    GSVD4PY_LAPACK environment variable, which must be set before the first
    call into gsvd4py.

    Returns
    -------
    info : dict
        ``lib_type``
            ``'accelerate'``, ``'scipy_openblas'`` or ``'system'`` -- the
            symbol spelling and calling convention in use.
        ``path``
            Path of the loaded library, or None when the symbols were found
            among those already loaded into the process.
        ``source``
            ``'override'`` if chosen via GSVD4PY_LAPACK, else ``'default'``.
        ``hidden_lengths``
            True when the gfortran hidden character-length arguments are
            passed.
        ``int_width``
            Width in bits of LAPACK integer arguments (always 32).
        ``scipy_lapack``
            The LAPACK provider SciPy reports, if detected, else None.

    Raises
    ------
    ImportError
        If no suitable LAPACK library can be loaded.
    """
    _load_lib()
    return {
        'lib_type': _lib_type,
        'path': _lib_path,
        'source': _lib_source,
        'hidden_lengths': _lib_type != 'accelerate',
        'int_width': 32,
        'scipy_lapack': None,
    }


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
        uses_hidden_lengths = False
    elif _lib_type == 'scipy_openblas':
        sym = f'scipy_{dtype_char}{base_name}_'
        uses_hidden_lengths = True
    else:   # 'system'
        sym = f'{dtype_char}{base_name}_'
        uses_hidden_lengths = True

    fn = _lib[sym]
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
