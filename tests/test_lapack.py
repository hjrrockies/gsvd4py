"""
Tests for LAPACK discovery (gsvd4py._lapack).

Validates:
  - The real environment loads a LAPACK library and reports it
  - Strategy probe order, with the strategies replaced by fakes (runs anywhere)
  - Detection of SciPy's LAPACK provider, and the order it selects
  - GSVD4PY_LAPACK: provider names, absolute paths, and invalid values
  - An explicitly requested provider that cannot load raises, never falls back
  - Accelerate and scipy_openblas32 agree numerically (macOS, when installed)
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import gsvd4py
import gsvd4py._lapack as _lapack


_LIB_TYPES = ('accelerate', 'scipy_openblas', 'system')
_INFO_KEYS = {'lib_type', 'path', 'source', 'hidden_lengths', 'int_width',
              'scipy_lapack'}
_CACHE_ATTRS = ('_lib', '_lib_type', '_lib_path', '_lib_source',
                '_scipy_lapack')
_REAL_DETECT = _lapack._detect_scipy_lapack
_LIB_EXT = _lapack._shared_lib_pattern()[1:].rstrip('*')   # .so .dylib .dll


def _reset_cache(monkeypatch):
    for attr in _CACHE_ATTRS:
        monkeypatch.setattr(_lapack, attr, None)
    monkeypatch.delenv('GSVD4PY_LAPACK', raising=False)


@pytest.fixture
def fresh_real(monkeypatch):
    """Clear the loaded-library cache and GSVD4PY_LAPACK for one test.

    monkeypatch restores the original cache afterwards, so the rest of the
    suite keeps using the library it already loaded.
    """
    _reset_cache(monkeypatch)
    return monkeypatch


@pytest.fixture
def fresh(fresh_real):
    """Like fresh_real, with SciPy detection stubbed out to 'unknown'.

    Keeps probe orders independent of the SciPy the tests happen to run on.
    """
    fresh_real.setattr(_lapack, '_detect_scipy_lapack', lambda: None)
    return fresh_real


def _fake_strategies(monkeypatch, succeed=()):
    """Replace every strategy with a fake; return the list of names called."""
    calls = []
    for name in list(_lapack._STRATEGIES):
        def fake(attempts, name=name):
            calls.append(name)
            if name in succeed:
                lib_type = 'accelerate' if name == 'accelerate' else 'system'
                return object(), lib_type, f'/fake/{name}'
            attempts.append(f'{name}: fake miss')
            return None
        monkeypatch.setitem(_lapack._STRATEGIES, name, fake)
    return calls


# ---------------------------------------------------------------------------
# Test: the real environment
# ---------------------------------------------------------------------------

class TestLibraryLoading:
    def test_loads_without_error(self):
        _lapack._load_lib()
        assert _lapack._lib_type in _LIB_TYPES

    def test_lapack_info_keys(self):
        info = gsvd4py.lapack_info()
        assert set(info) == _INFO_KEYS
        assert info['lib_type'] in _LIB_TYPES
        assert info['source'] in ('override', 'detected', 'default')
        assert info['scipy_lapack'] in ('accelerate', 'openblas', 'other',
                                        None)
        assert info['int_width'] == 32

    def test_lapack_info_matches_calling_convention(self):
        info = gsvd4py.lapack_info()
        _, uses_hidden_lengths = _lapack.get_ggsvd3('d')
        assert info['lib_type'] == _lapack._lib_type
        assert info['hidden_lengths'] == uses_hidden_lengths

    @pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')
    def test_accelerate_scipy_keeps_accelerate(self, fresh_real):
        # The pip-on-macOS-14+ default must not change.
        if _REAL_DETECT() != 'accelerate':
            pytest.skip('this SciPy is not built against Accelerate')
        info = gsvd4py.lapack_info()
        assert info['lib_type'] == 'accelerate'
        assert info['source'] == 'detected'
        assert info['scipy_lapack'] == 'accelerate'


# ---------------------------------------------------------------------------
# Test: default probe order (fake strategies)
# ---------------------------------------------------------------------------

class TestProbeOrder:
    def test_default_order_is_tried_in_full(self, fresh):
        calls = _fake_strategies(fresh)
        with pytest.raises(ImportError, match='Could not find a LAPACK'):
            _lapack._load_lib()
        assert calls == list(_lapack._DEFAULT_ORDER)

    def test_error_lists_every_attempt(self, fresh):
        _fake_strategies(fresh)
        with pytest.raises(ImportError) as excinfo:
            _lapack._load_lib()
        for name in _lapack._DEFAULT_ORDER:
            assert f'{name}: fake miss' in str(excinfo.value)

    def test_stops_at_first_success(self, fresh):
        calls = _fake_strategies(fresh, succeed={'scipy_openblas32',
                                                 'find_library'})
        _lapack._load_lib()
        stop = _lapack._DEFAULT_ORDER.index('scipy_openblas32')
        assert calls == list(_lapack._DEFAULT_ORDER[:stop + 1])
        assert _lapack._lib_path == '/fake/scipy_openblas32'
        assert _lapack._lib_source == 'default'

    def test_cached_after_first_load(self, fresh):
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        _lapack._load_lib()
        _lapack._load_lib()
        assert len(calls) == 1

    @pytest.mark.parametrize('preferred',
                             [None, 'accelerate', 'openblas', 'other'])
    def test_every_order_contains_every_strategy(self, preferred):
        order = _lapack._probe_order(preferred)
        assert sorted(order) == sorted(_lapack._STRATEGIES)


# ---------------------------------------------------------------------------
# Test: probe order follows SciPy's detected provider (fake strategies)
# ---------------------------------------------------------------------------

class TestDetectedOrder:
    @pytest.mark.parametrize('preferred', [None, 'accelerate'])
    def test_accelerate_or_unknown_keeps_default(self, preferred):
        assert _lapack._probe_order(preferred) == list(_lapack._DEFAULT_ORDER)

    def test_openblas_prefers_scipy_openblas(self):
        order = _lapack._probe_order('openblas')
        assert order[:2] == ['scipy_bundled', 'scipy_openblas32']
        assert order[-1] == 'accelerate'

    def test_other_prefers_the_environment(self):
        order = _lapack._probe_order('other')
        assert order[0] == 'env_prefix'
        assert order[-1] == 'accelerate'

    @pytest.mark.parametrize('preferred', ['accelerate', 'openblas', 'other'])
    def test_load_follows_detection(self, fresh, preferred):
        fresh.setattr(_lapack, '_detect_scipy_lapack', lambda: preferred)
        calls = _fake_strategies(fresh)
        with pytest.raises(ImportError):
            _lapack._load_lib()
        assert calls == _lapack._probe_order(preferred)

    def test_detected_source_is_reported(self, fresh):
        fresh.setattr(_lapack, '_detect_scipy_lapack', lambda: 'openblas')
        calls = _fake_strategies(fresh, succeed={'scipy_bundled',
                                                 'accelerate'})
        info = gsvd4py.lapack_info()
        assert calls == ['scipy_bundled']
        assert info['source'] == 'detected'
        assert info['scipy_lapack'] == 'openblas'

    def test_detector_that_raises_uses_default(self, fresh):
        def boom():
            raise RuntimeError('detection blew up')
        fresh.setattr(_lapack, '_detect_scipy_lapack', boom)
        calls = _fake_strategies(fresh)
        with pytest.raises(ImportError, match='Could not find a LAPACK'):
            _lapack._load_lib()
        assert calls == list(_lapack._DEFAULT_ORDER)
        assert _lapack._scipy_lapack is None

    def test_override_beats_detection(self, fresh):
        fresh.setattr(_lapack, '_detect_scipy_lapack', lambda: 'openblas')
        fresh.setenv('GSVD4PY_LAPACK', 'accelerate')
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        info = gsvd4py.lapack_info()
        assert calls == ['accelerate']
        assert info['source'] == 'override'
        assert info['scipy_lapack'] == 'openblas'


# ---------------------------------------------------------------------------
# Test: detecting SciPy's LAPACK provider
# ---------------------------------------------------------------------------

def _config(name):
    def show_config(mode='stdout'):
        assert mode == 'dicts'
        return {'Build Dependencies': {'lapack': {'name': name}}}
    return show_config


class TestDetectScipyLapack:
    @pytest.fixture
    def scipy_mod(self, monkeypatch, tmp_path):
        """SciPy with an empty bundled-library directory."""
        scipy = pytest.importorskip('scipy')
        monkeypatch.setattr(_lapack, '_scipy_bundled_lib_dirs',
                            lambda: [str(tmp_path)])
        return scipy

    def test_bundled_openblas_wins(self, scipy_mod, monkeypatch, tmp_path):
        (tmp_path / f'libscipy_openblas{_LIB_EXT}').write_bytes(b'')
        monkeypatch.setattr(scipy_mod, 'show_config', _config('Accelerate'))
        assert _REAL_DETECT() == 'openblas'

    def test_bundled_fortran_runtime_is_not_openblas(self, scipy_mod,
                                                     monkeypatch, tmp_path):
        # Accelerate wheels vendor libgfortran too.
        (tmp_path / f'libgfortran.5{_LIB_EXT}').write_bytes(b'')
        monkeypatch.setattr(scipy_mod, 'show_config', _config('Accelerate'))
        assert _REAL_DETECT() == 'accelerate'

    @pytest.mark.parametrize('name, expected', [
        ('Accelerate', 'accelerate'),
        ('scipy-openblas', 'openblas'),
        ('openblas', 'openblas'),
        ('lapack', 'other'),
        ('mkl', 'other'),
    ])
    def test_build_config_name(self, scipy_mod, monkeypatch, name, expected):
        monkeypatch.setattr(scipy_mod, 'show_config', _config(name))
        assert _REAL_DETECT() == expected

    def test_old_scipy_without_dicts_mode(self, scipy_mod, monkeypatch):
        def show_config():
            print('an old-style config dump')
        monkeypatch.setattr(scipy_mod, 'show_config', show_config)
        assert _REAL_DETECT() is None

    def test_config_returning_none(self, scipy_mod, monkeypatch):
        monkeypatch.setattr(scipy_mod, 'show_config', lambda mode=None: None)
        assert _REAL_DETECT() is None

    def test_config_output_is_silenced(self, scipy_mod, monkeypatch, capsys):
        def show_config(mode='stdout'):
            print('noise')
            return _config('Accelerate')(mode)
        monkeypatch.setattr(scipy_mod, 'show_config', show_config)
        assert _REAL_DETECT() == 'accelerate'
        assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# Test: the environment-prefix strategy
# ---------------------------------------------------------------------------

class TestEnvPrefix:
    def _lib_dir(self, prefix):
        if sys.platform == 'win32':
            return prefix / 'Library' / 'bin'
        return prefix / 'lib'

    def test_prefix_without_lapack(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'prefix', str(tmp_path))
        attempts = []
        assert _lapack._strategy_env_prefix(attempts) is None
        assert attempts == [f"{self._lib_dir(tmp_path)}: no LAPACK libraries"]

    def test_candidates_are_tried_in_preference_order(self, monkeypatch,
                                                      tmp_path):
        lib_dir = self._lib_dir(tmp_path)
        lib_dir.mkdir(parents=True)
        for name in ('libopenblas', 'liblapack', 'libz'):
            (lib_dir / f'{name}{_LIB_EXT}').write_bytes(b'not a library')
        monkeypatch.setattr(sys, 'prefix', str(tmp_path))
        attempts = []
        assert _lapack._strategy_env_prefix(attempts) is None
        assert len(attempts) == 2
        assert f'liblapack{_LIB_EXT}: could not load' in attempts[0]
        assert f'libopenblas{_LIB_EXT}: could not load' in attempts[1]


# ---------------------------------------------------------------------------
# Test: GSVD4PY_LAPACK override (fake strategies)
# ---------------------------------------------------------------------------

class TestOverride:
    @pytest.mark.parametrize('provider', sorted(_lapack._OVERRIDE_GROUPS))
    def test_provider_uses_only_its_strategies(self, fresh, provider):
        fresh.setenv('GSVD4PY_LAPACK', provider)
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        _lapack._load_lib()
        assert calls == [_lapack._OVERRIDE_GROUPS[provider][0]]
        assert _lapack._lib_source == 'override'

    @pytest.mark.parametrize('provider', sorted(_lapack._OVERRIDE_GROUPS))
    def test_unavailable_provider_raises_without_fallback(self, fresh,
                                                         provider):
        fresh.setenv('GSVD4PY_LAPACK', provider)
        group = _lapack._OVERRIDE_GROUPS[provider]
        others = set(_lapack._STRATEGIES) - set(group)
        calls = _fake_strategies(fresh, succeed=others)
        with pytest.raises(ImportError, match='was requested'):
            _lapack._load_lib()
        assert calls == list(group)
        assert _lapack._lib is None

    def test_name_is_case_and_space_insensitive(self, fresh):
        fresh.setenv('GSVD4PY_LAPACK', '  System ')
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        _lapack._load_lib()
        assert calls == [_lapack._OVERRIDE_GROUPS['system'][0]]

    def test_empty_value_means_unset(self, fresh):
        fresh.setenv('GSVD4PY_LAPACK', '')
        calls = _fake_strategies(fresh, succeed={'accelerate'})
        _lapack._load_lib()
        assert calls == ['accelerate']
        assert _lapack._lib_source == 'default'

    @pytest.mark.parametrize('value', ['openblas', 'mkl', 'liblapack.so'])
    def test_invalid_value_raises(self, fresh, value):
        fresh.setenv('GSVD4PY_LAPACK', value)
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        with pytest.raises(ImportError, match='invalid GSVD4PY_LAPACK'):
            _lapack._load_lib()
        assert calls == []

    def test_missing_path_raises(self, fresh, tmp_path):
        missing = str(tmp_path / 'no_such_lapack.so')
        fresh.setenv('GSVD4PY_LAPACK', missing)
        calls = _fake_strategies(fresh, succeed=set(_lapack._STRATEGIES))
        with pytest.raises(ImportError, match='does not provide dggsvd3'):
            _lapack._load_lib()
        assert calls == []

    def test_lapack_info_reports_override(self, fresh):
        fresh.setenv('GSVD4PY_LAPACK', 'accelerate')
        _fake_strategies(fresh, succeed={'accelerate'})
        info = gsvd4py.lapack_info()
        assert info['source'] == 'override'
        assert info['lib_type'] == 'accelerate'
        assert info['hidden_lengths'] is False


# ---------------------------------------------------------------------------
# Test: GSVD4PY_LAPACK with a real library path
# ---------------------------------------------------------------------------

def _scipy_openblas32_lib():
    scipy_openblas32 = pytest.importorskip('scipy_openblas32')
    paths = glob.glob(os.path.join(scipy_openblas32.get_lib_dir(),
                                   _lapack._shared_lib_pattern()))
    if not paths:
        pytest.skip('scipy_openblas32 ships no shared library here')
    return paths[0]


class TestOverridePath:
    def test_path_to_scipy_openblas32(self, fresh):
        path = _scipy_openblas32_lib()
        fresh.setenv('GSVD4PY_LAPACK', path)
        info = gsvd4py.lapack_info()
        assert info == dict(info, lib_type='scipy_openblas', path=path,
                            source='override', hidden_lengths=True)

    def test_gsvdvals_through_path_override(self, fresh):
        path = _scipy_openblas32_lib()
        fresh.setenv('GSVD4PY_LAPACK', path)
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 4))
        B = rng.standard_normal((5, 4))
        c, s = gsvd4py.gsvdvals(A, B)
        np.testing.assert_allclose(c**2 + s**2, 1, rtol=1e-12)
        assert _lapack._lib_type == 'scipy_openblas'


# ---------------------------------------------------------------------------
# Test: Accelerate and scipy_openblas32 agree (one subprocess per provider)
# ---------------------------------------------------------------------------

_CROSS_PROVIDER_SCRIPT = r'''
import json, sys
import numpy as np
import gsvd4py
from gsvd4py import gsvd, gsvdvals

def pair(case, dtype, rng):
    def rand(*shape):
        x = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            x = x + 1j * rng.standard_normal(shape)
        return x.astype(dtype)
    if case == 'general':
        return rand(8, 5), rand(6, 5)
    if case == 'rank_deficient':      # stacked rank 3 < p = 5, exactly
        W = rand(3, 5)
        return rand(8, 3) @ W, rand(6, 3) @ W
    if case == 'm_lt_kl':             # m = 2 < k + l = 5
        return rand(2, 5), rand(3, 5)

out = {'info': gsvd4py.lapack_info(), 'results': {}}
for case in ('general', 'rank_deficient', 'm_lt_kl'):
    for dtype in (np.float64, np.complex128):
        A, B = pair(case, dtype, np.random.default_rng(1234))
        c, s = gsvdvals(A, B)
        U, V, C, S, X = gsvd(A, B)
        *_, k, l = gsvd(A, B, mode='separate')
        XH = X.conj().T
        out['results'][case + '/' + np.dtype(dtype).name] = {
            'c': c.tolist(), 's': s.tolist(), 'k': int(k), 'l': int(l),
            'res_a': float(np.linalg.norm(A - U @ C @ XH) / np.linalg.norm(A)),
            'res_b': float(np.linalg.norm(B - V @ S @ XH) / np.linalg.norm(B)),
        }
json.dump(out, sys.stdout)
'''


def _run_with_provider(provider):
    env = dict(os.environ, GSVD4PY_LAPACK=provider)
    proc = subprocess.run([sys.executable, '-c', _CROSS_PROVIDER_SCRIPT],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f'GSVD4PY_LAPACK={provider} run failed '
                    f'(exit {proc.returncode}):\n{proc.stderr}')
    return json.loads(proc.stdout)


@pytest.mark.skipif(sys.platform != 'darwin', reason='needs Accelerate')
class TestCrossProvider:
    def test_accelerate_and_openblas_agree(self):
        pytest.importorskip('scipy_openblas32')
        acc = _run_with_provider('accelerate')
        obl = _run_with_provider('scipy_openblas')
        assert acc['info']['lib_type'] == 'accelerate'
        assert obl['info']['lib_type'] == 'scipy_openblas'
        assert acc['results'].keys() == obl['results'].keys()

        # U, V and X are not unique; compare what is.
        for key, a in acc['results'].items():
            o = obl['results'][key]
            assert (a['k'], a['l']) == (o['k'], o['l']), key
            np.testing.assert_allclose(a['c'], o['c'], atol=1e-10,
                                       err_msg=key)
            np.testing.assert_allclose(a['s'], o['s'], atol=1e-10,
                                       err_msg=key)
            for r in (a, o):
                assert r['res_a'] < 1e-11, key
                assert r['res_b'] < 1e-11, key
