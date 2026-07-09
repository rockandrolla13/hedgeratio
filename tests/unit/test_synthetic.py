"""Phase 1/4 tests for the synthetic DGPs (S1-S4 implemented)."""
import numpy as np
import pytest

from hrl.data.synthetic import _REGISTRY, SyntheticSample, generate

IMPLEMENTED = ["S1", "S2", "S3", "S4", "S7", "S7b"]
STUBBED = ["S5", "S6"]


def test_registry_covers_all_dgps():
    """Every DGP name dispatches."""
    assert set(_REGISTRY) == set(IMPLEMENTED) | set(STUBBED)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_shapes_and_alignment(name):
    """Each implemented DGP returns aligned, correctly shaped arrays."""
    n = 500
    s = generate(name, n_steps=n, seed=0)
    assert isinstance(s, SyntheticSample)
    for arr in (s.y, s.x, s.beta_true, s.alpha_true):
        assert arr.shape == (n,)
    assert np.all(np.isfinite(s.y)) and np.all(np.isfinite(s.x))


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_determinism(name):
    """Same seed reproduces the same sample exactly."""
    a = generate(name, n_steps=300, seed=7)
    b = generate(name, n_steps=300, seed=7)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.x, b.x)


def test_s1_static_beta_is_constant():
    """S1 truth is a flat beta."""
    s = generate("S1", n_steps=400, seed=0)
    assert np.ptp(s.beta_true) == 0.0


def test_s2_beta_drifts():
    """S2 truth varies (genuine drift, no breaks)."""
    s = generate("S2", n_steps=1000, seed=0)
    assert np.ptp(s.beta_true) > 0.1
    assert s.break_times.size == 0


def test_s3_marks_outliers():
    """S3 records the contamination indices."""
    s = generate("S3", n_steps=2000, seed=0)
    assert s.outlier_times.size > 0


def test_s4_exposes_regime_labels():
    """S4 returns 0/1 regime labels with both calm and stress windows, constant true beta."""
    s = generate("S4", n_steps=2000, seed=0)
    assert s.regime.shape == (2000,)
    assert set(np.unique(s.regime)).issubset({0, 1})
    assert s.regime.min() == 0 and s.regime.max() == 1     # both regimes occur
    assert np.ptp(s.beta_true) == 0.0                       # beta is constant in S4


@pytest.mark.parametrize("name", STUBBED)
def test_stubbed_dgps_raise(name):
    """S5-S6 are not implemented yet and say so."""
    with pytest.raises(NotImplementedError):
        generate(name, n_steps=100, seed=0)
