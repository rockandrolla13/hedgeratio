"""Mechanism 1 deterministic + efficiency tests (spec T1, T4).

T1 (influence saturation, deterministic): vanilla update is linear in the innovation (R^2>0.999),
WoLF-IMQ influence peaks at |e| = c*sqrt(S) and strictly decreases beyond 2c*sqrt(S), and the
Student-t reference weight yields bounded influence far below vanilla at large |e|.
T4 (efficiency under the null): on clean S1, relative efficiency stays near 1.

The T1 single-step harness uses the hph << R regime (tiny P, unit R) so S ~ R and the influence
peak sits at the spec's c*sqrt(S); with comparable hph the peak merely shifts, which is regime
detail, not the phenomenon under test.
"""
import numpy as np

from hrl.config import PipelineConfig
from hrl.core.context import StepContext
from hrl.data.synthetic import generate
from hrl.eval import metrics, stats
from hrl.filters.pipeline import Pipeline
from hrl.filters.stages.gaussian import GaussianUpdateStage
from hrl.filters.stages.wolf import IMQWeight, StudentTWeight
from hrl.filters.student_t import StudentTKalmanRW
from hrl.models.linear_ssm import LinearGaussianSSM

_MODEL = LinearGaussianSSM()
_P, _R = 1e-4, 1.0                    # hph = 2*_P << _R  =>  S ~= _R, peak at c*sqrt(S)


def _influence(weight_fn, e: float) -> tuple[float, float]:
    """||theta_post - theta_pred|| and S for a single update with innovation exactly e."""
    theta_pred = np.array([0.0, 1.0])
    ctx = StepContext(t=0, y=1.0 + float(e), x=1.0, theta=theta_pred.copy(),
                      P=np.eye(2) * _P, R=_R)
    GaussianUpdateStage(weight_fn=weight_fn).apply(ctx, _MODEL)
    return float(np.linalg.norm(ctx.theta - theta_pred)), ctx.S


def test_t1_vanilla_influence_is_linear():
    """Vanilla state shift is an (essentially perfect) linear function of the innovation."""
    _, S = _influence(None, 0.0)
    es = np.linspace(0.0, 50.0 * np.sqrt(S), 4000)
    infl = np.array([_influence(None, e)[0] for e in es])
    A = np.vstack([es, np.ones_like(es)]).T
    coef, *_ = np.linalg.lstsq(A, infl, rcond=None)
    r2 = 1.0 - np.sum((infl - A @ coef) ** 2) / np.sum((infl - infl.mean()) ** 2)
    assert r2 > 0.999


def test_t1_wolf_peaks_at_c_root_s_and_redescends():
    """WoLF-IMQ influence peaks near |e| = c*sqrt(S) and strictly decreases past 2c*sqrt(S)."""
    c = 3.0
    imq = IMQWeight(c=c)
    _, S = _influence(imq, 0.0)
    rs = np.sqrt(S)
    es = np.linspace(0.0, 20.0 * rs, 8000)
    infl = np.array([_influence(imq, e)[0] for e in es])
    peak_e = es[int(np.argmax(infl))]
    assert 0.85 <= peak_e / (c * rs) <= 1.15                 # peak at ~ c*sqrt(S)
    i = lambda e: _influence(imq, e)[0]
    assert i(3.0 * c * rs) < i(2.0 * c * rs)                 # strictly decreasing beyond 2c*sqrtS
    assert i(6.0 * c * rs) < i(3.0 * c * rs)
    assert _influence(imq, 200.0 * rs)[1] > 0.0             # S well defined at extreme e


def test_t1_student_t_reference_is_bounded():
    """The Student-t reference weight keeps single-step influence bounded and far below vanilla."""
    st = StudentTWeight(nu=4.0)
    _, S = _influence(st, 0.0)
    rs = np.sqrt(S)
    es = np.linspace(0.0, 50.0 * rs, 4000)
    infl = np.array([_influence(st, e)[0] for e in es])
    peak = infl.max()
    van_far, _ = _influence(None, 50.0 * rs)
    assert peak < 5.0 * infl[np.searchsorted(es, rs)]        # no blow-up: bounded influence
    assert infl[-1] < 0.05 * van_far                          # far below the linear vanilla shift


def test_t4_efficiency_under_null_s1():
    """RE <= 1.10 for WoLF (default c) and <= 1.05 for the Student-t filter (nu=8) on clean S1."""
    n_paths, n, r = 100, 1500, 4e-4

    def pipe(**kw):
        cfg = PipelineConfig(noise_model="fixed", r=r, p0=10.0, **kw)
        return Pipeline.from_config(cfg, LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5))

    van = np.empty(n_paths)
    wolf = np.empty(n_paths)
    tfilt = np.empty(n_paths)
    for i in range(n_paths):
        s = generate("S1", n_steps=n, seed=7000 + i)
        van[i] = metrics.beta_rmse(pipe(weight_fn="none").run(s.y, s.x).beta, s.beta_true)
        wolf[i] = metrics.beta_rmse(pipe(weight_fn="imq", c=3.0).run(s.y, s.x).beta, s.beta_true)
        tfilt[i] = metrics.beta_rmse(
            StudentTKalmanRW(nu=8.0, r=r, p0=10.0).run(s.y, s.x).beta, s.beta_true)
    assert stats.relative_efficiency(wolf, van) <= 1.10
    assert stats.relative_efficiency(tfilt, van) <= 1.05
