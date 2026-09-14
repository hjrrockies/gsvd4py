"""
Tests for LAPACK discovery (gsvd4py._lapack).

Validates:
  - The real environment loads a LAPACK library and reports it
  - Strategy probe order, with the strategies replaced by fakes (runs anywhere)
  - GSVD4PY_LAPACK: provider names, absolute paths, and invalid values
  - An explicitly requested provider that cannot load raises, never falls back
"""

import numpy as np
import pytest

import gsvd4py
import gsvd4py._lapack as _lapack


_LIB_TYPES = ('accelerate', 'scipy_openblas', 'system')
_INFO_KEYS = {'lib_type', 'path', 'source', 'hidden_lengths', 'int_width',
              'scipy_lapack'}


@pytest.fixture
def fresh(monkeypatch):
    """Clear the loaded-library cache and GSVD4PY_LAPACK for one test.

    monkeypatch restores the original cache afterwards, so the rest of the
    suite keeps using the library it already loaded.
    """
    for attr in ('_lib', '_lib_type', '_lib_path', '_lib_source'):
        monkeypatch.setattr(_lapack, attr, None)
    monkeypatch.delenv('GSVD4PY_LAPACK', raising=False)
    return monkeypatch


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
        assert info['source'] in ('override', 'default')
        assert info['int_width'] == 32

    def test_lapack_info_matches_calling_convention(self):
        info = gsvd4py.lapack_info()
        _, uses_hidden_lengths = _lapack.get_ggsvd3('d')
        assert info['lib_type'] == _lapack._lib_type
        assert info['hidden_lengths'] == uses_hidden_lengths


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

    def test_default_order_covers_every_strategy(self):
        assert sorted(_lapack._DEFAULT_ORDER) == sorted(_lapack._STRATEGIES)


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
        assert calls == ['process_symbols']

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
    import glob
    import os
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
