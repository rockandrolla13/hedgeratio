"""Tests for the Student-t scale-mixture measurement filter (Mechanism 1.2)."""
import numpy as np

from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.baselines import VanillaKalmanRW
from hrl.filters.student_t import StudentTKalmanRW


def test_recovers_static_beta_on_s1():
    """On clean S1 the t-filter converges to the true constant hedge ratio."""
    s = generate("S1", n_steps=1500, seed=0)
    res = StudentTKalmanRW(nu=5.0, r=4e-4, p0=10.0).run(s.y, s.x)
    assert abs(res.beta[-1] - s.beta_true[-1]) < 0.05


def test_large_nu_approaches_gaussian():
    """As nu -> large the scale mixture collapses to the Gaussian (vanilla) filter."""
    s = generate("S1", n_steps=800, seed=1)
    tf = StudentTKalmanRW(nu=1e6, q_alpha=1e-7, q_beta=1e-5, r=1e-3, p0=1.0).run(s.y, s.x)
    van = VanillaKalmanRW(q_alpha=1e-7, q_beta=1e-5, r=1e-3, p0=1.0).run(s.y, s.x)
    assert np.allclose(tf.beta, van.beta, atol=1e-4)


def test_weights_are_lambda_bounded_and_bite_at_outliers():
    """Converged lambda_t lies in (0, (nu+1)/nu] and is smallest at the injected outliers.

    lambda is a precision multiplier: it exceeds 1 (up-weighting) for sub-nominal residuals and
    tends to 0 for extreme ones, so its upper bound is (nu+1)/nu, not 1.
    """
    nu = 4.0
    s = generate("S3", n_steps=1500, seed=3)
    res = StudentTKalmanRW(nu=nu, r=4e-4, p0=10.0).run(s.y, s.x)
    assert np.all((res.weights > 0.0) & (res.weights <= (nu + 1.0) / nu + 1e-9))
    if s.outlier_times.size:
        assert np.median(res.weights[s.outlier_times]) < np.median(res.weights)


def test_beats_vanilla_on_heavy_tails():
    """On heavy-tailed S3 the t-filter's beta RMSE beats the vanilla KF on average."""
    n_paths, n, r = 30, 1500, 4e-4
    van = np.empty(n_paths)
    tf = np.empty(n_paths)
    for i in range(n_paths):
        s = generate("S3", n_steps=n, seed=5000 + i)
        van[i] = metrics.beta_rmse(
            VanillaKalmanRW(q_alpha=1e-7, q_beta=1e-5, r=r, p0=10.0).run(s.y, s.x).beta,
            s.beta_true)
        tf[i] = metrics.beta_rmse(
            StudentTKalmanRW(nu=4.0, r=r, p0=10.0).run(s.y, s.x).beta, s.beta_true)
    assert tf.mean() < van.mean()
