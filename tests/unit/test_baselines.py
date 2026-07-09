"""Phase 1 tests for the baselines and the qualitative KF-vs-OLS result."""
import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.baselines import (
    VanillaKalmanRW,
    ewma_half_life,
    rolling_ols,
    rolling_tls,
    steady_state_gain,
)
from hrl.filters.pipeline import Pipeline
from hrl.models.linear_ssm import LinearGaussianSSM


def test_import():
    """Smoke: baseline callables and class are importable/constructible."""
    assert callable(rolling_ols)
    assert callable(rolling_tls)
    assert VanillaKalmanRW() is not None


def test_rolling_ols_recovers_static_beta():
    """On S1, rolling OLS recovers the true (constant) hedge ratio in the bulk."""
    s = generate("S1", n_steps=1500, seed=0)
    beta = rolling_ols(s.y, s.x, window=120)
    valid = beta[~np.isnan(beta)]
    assert abs(np.median(valid) - s.beta_true[-1]) < 0.1


def test_rolling_tls_recovers_static_beta():
    """Rolling TLS also recovers the static beta."""
    s = generate("S1", n_steps=1500, seed=0)
    beta = rolling_tls(s.y, s.x, window=120)
    valid = beta[~np.isnan(beta)]
    assert abs(np.median(valid) - s.beta_true[-1]) < 0.1


def test_kf_less_churn_than_rolling_ols():
    """Acceptance #1 (qualitative): the KF hedge ratio churns far less than rolling OLS."""
    s = generate("S1", n_steps=2000, seed=0)
    model = LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5)
    cfg = PipelineConfig(weight_fn="none", noise_model="fixed", r=4e-4, p0=10.0)
    kf_beta = Pipeline.from_config(cfg, model).run(s.y, s.x).beta

    ols_beta = rolling_ols(s.y, s.x, window=120)
    ols_beta = ols_beta[~np.isnan(ols_beta)]

    assert metrics.delta_beta_var(kf_beta[-len(ols_beta):]) < metrics.delta_beta_var(ols_beta)


# --- Phase 2: EM/MLE fit and EWRLS equivalence -------------------------------------------

def test_ewma_half_life_monotonic():
    """Higher process/observation-noise ratio => shorter effective memory."""
    assert 0.0 < steady_state_gain(1e-5, 1.0) < 1.0
    hl_slow = ewma_half_life(1e-6, 1.0)
    hl_fast = ewma_half_life(1e-3, 1.0)
    assert hl_fast < hl_slow
    assert ewma_half_life(0.0, 1.0) == float("inf")


def test_fit_improves_likelihood():
    """Fitting (Q, R) does not worsen the training prediction log-likelihood."""
    s = generate("S1", n_steps=1500, seed=0)
    start = VanillaKalmanRW(q_alpha=1e-4, q_beta=1e-2, r=1e-1)
    nll_before = start.neg_log_likelihood(s.y, s.x)
    fitted = VanillaKalmanRW(q_alpha=1e-4, q_beta=1e-2, r=1e-1).fit(s.y, s.x)
    nll_after = fitted.neg_log_likelihood(s.y, s.x)
    assert nll_after <= nll_before + 1e-6


def test_fit_adapts_q_to_drift():
    """Validation vs S1/S2: the fitted beta process-noise is larger on drifting data."""
    s1 = generate("S1", n_steps=2000, seed=0)
    s2 = generate("S2", n_steps=2000, seed=0)
    q_static = VanillaKalmanRW(q_alpha=1e-6, q_beta=1e-4, r=1e-2).fit(s1.y, s1.x).q_beta
    q_drift = VanillaKalmanRW(q_alpha=1e-6, q_beta=1e-4, r=1e-2).fit(s2.y, s2.x).q_beta
    assert q_drift > q_static


def test_fit_tracks_drift_walkforward():
    """Fit on the first 40% of S2, then the fitted KF tracks the held-out drift well."""
    s = generate("S2", n_steps=2500, seed=1)
    cut = int(0.4 * len(s.y))
    kf = VanillaKalmanRW(q_alpha=1e-6, q_beta=1e-4, r=1e-2).fit(s.y[:cut], s.x[:cut])
    res = kf.run(s.y, s.x)
    oos = slice(cut, None)
    assert metrics.beta_rmse(res.beta[oos], s.beta_true[oos]) < 0.15
