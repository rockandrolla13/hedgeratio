"""Phase 5 (groundwork) tests for the slow hedge-ratio anchors.

Cointegrated test data is generated inline here (not via data/synthetic.py) so these tests are
independent of the DGP registry.
"""
import numpy as np

from hrl.anchors.johansen import JohansenAnchor, johansen_vector
from hrl.anchors.tls import TlsAnchor, tls_fit


def _cointegrated(n, beta=1.2, alpha=0.5, seed=0, noise_sd=0.05, rho=0.8):
    """x = random walk; spread = AR(1) stationary; y = alpha + beta x + spread."""
    rng = np.random.default_rng(seed)
    x = 4.6 + np.cumsum(rng.normal(0, 0.02, n))
    spread = np.empty(n)
    spread[0] = 0.0
    for t in range(1, n):
        spread[t] = rho * spread[t - 1] + rng.normal(0, noise_sd)
    y = alpha + beta * x + spread
    return y, x


def _errors_in_variables(n, beta=1.2, seed=0, sd=0.1):
    """Latent x*; observe x = x* + noise, y = beta x* + noise (equal noise variances)."""
    rng = np.random.default_rng(seed)
    xstar = np.cumsum(rng.normal(0, 0.5, n))
    x = xstar + rng.normal(0, sd, n)
    y = beta * xstar + rng.normal(0, sd, n)
    return y, x


# --- point estimators --------------------------------------------------------------------

def test_johansen_recovers_beta():
    """Johansen recovers the true cointegrating hedge ratio."""
    y, x = _cointegrated(1500, beta=1.2, seed=1)
    assert abs(johansen_vector(y, x) - 1.2) < 0.1


def test_tls_recovers_beta():
    """TLS recovers the true hedge ratio on cointegrated data."""
    y, x = _cointegrated(1500, beta=1.2, seed=1)
    assert abs(tls_fit(y, x) - 1.2) < 0.1


def test_tls_beats_ols_under_eiv():
    """Under errors-in-variables, TLS is less biased than OLS(y|x) (attenuation)."""
    y, x = _errors_in_variables(3000, beta=1.2, seed=2, sd=0.15)
    ols = np.polyfit(x, y, 1)[0]
    tls = tls_fit(y, x)
    assert abs(tls - 1.2) < abs(ols - 1.2)


def test_degenerate_inputs_return_nan():
    """Degenerate windows return nan rather than raising."""
    assert np.isnan(tls_fit(np.array([1.0]), np.array([1.0])))
    assert np.isnan(johansen_vector(np.zeros(5), np.zeros(5)))


# --- causal provider behaviour -----------------------------------------------------------

def test_anchor_nan_before_min_obs():
    """Both anchors return nan until min_obs observations are available."""
    y, x = _cointegrated(600, seed=3)
    for anchor in (TlsAnchor(min_obs=252), JohansenAnchor(min_obs=252)):
        assert np.isnan(anchor.anchor(100, y, x))
        assert np.isfinite(anchor.anchor(300, y, x))


def test_anchor_is_causal():
    """anchor(t) depends only on data strictly before t: mutating the future is a no-op."""
    y, x = _cointegrated(800, seed=4)
    a = TlsAnchor(window=252, recompute_every=21, min_obs=252)
    val = a.anchor(400, y, x)
    y2 = y.copy()
    y2[400:] += 100.0                       # corrupt the future
    b = TlsAnchor(window=252, recompute_every=21, min_obs=252)
    assert abs(b.anchor(400, y2, x) - val) < 1e-12


def test_anchor_blend_is_smooth():
    """Consecutive anchor values move gradually (no raw step jumps from recompute)."""
    y, x = _cointegrated(1500, beta=1.2, seed=5)
    a = TlsAnchor(window=252, recompute_every=21, blend_days=5, min_obs=252)
    vals = np.array([a.anchor(t, y, x) for t in range(1500)])
    vals = vals[np.isfinite(vals)]
    steps = np.abs(np.diff(vals))
    # no single-day jump should be large relative to the anchor scale
    assert np.nanmax(steps) < 0.15
