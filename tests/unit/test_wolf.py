"""Phase 3 tests for WoLF robust weighting."""
import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.pipeline import Pipeline
from hrl.filters.stages.gaussian import GaussianUpdateStage, PredictStage
from hrl.filters.stages.wolf import (
    HuberWeight,
    IMQWeight,
    StudentTWeight,
    WolfReweightStage,
)
from hrl.models.linear_ssm import LinearGaussianSSM


def _pipe(weight_fn, r=4e-4, c=3.0):
    model = LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5)
    cfg = PipelineConfig(weight_fn=weight_fn, noise_model="fixed", r=r, p0=10.0, c=c)
    return Pipeline.from_config(cfg, model)


def test_import_and_instantiate():
    """Smoke: weight functions and (deprecated) stage construct."""
    assert IMQWeight(c=3.0).c == 3.0
    assert HuberWeight() is not None
    assert StudentTWeight() is not None
    assert WolfReweightStage(IMQWeight()).name == "wolf"


def test_imq_matches_closed_form():
    """IMQ weight equals the hand-computed closed form (pins us to the paper's formula)."""
    # e=2, S=1, c=1 -> (1 + 4)^-0.5 = 1/sqrt(5)
    assert abs(IMQWeight(c=1.0).weight(2.0, 1.0) - 5.0 ** -0.5) < 1e-12
    # e=1, S=4, c=2 -> (1 + 1/16)^-0.5
    assert abs(IMQWeight(c=2.0).weight(1.0, 4.0) - (1.0625) ** -0.5) < 1e-12


def test_weights_in_unit_interval_and_monotone():
    """All weights lie in (0, 1] and shrink as the innovation grows."""
    for wf in (IMQWeight(c=3.0), HuberWeight(), StudentTWeight()):
        es = np.linspace(0.0, 50.0, 200)
        ws = np.array([wf.weight(e, 1.0) for e in es])
        assert np.all((ws > 0.0) & (ws <= 1.0 + 1e-12))
        assert ws[-1] < ws[0]                       # down-weights large innovations
    assert IMQWeight(c=3.0).weight(0.0, 1.0) == 1.0  # no down-weight at e=0


def test_c_infinity_recovers_vanilla():
    """c -> inf gives w == 1 and the WoLF pipeline equals the vanilla pipeline exactly."""
    assert IMQWeight(c=np.inf).weight(123.0, 2.0) == 1.0
    s = generate("S1", n_steps=800, seed=0)
    r_wolf = _pipe("imq", c=np.inf).run(s.y, s.x)
    r_van = _pipe("none").run(s.y, s.x)
    assert np.allclose(r_wolf.beta, r_van.beta, atol=1e-12)


def _state_shift(weight_fn, e, r=1e-3):
    """||theta_post - theta_pred|| and weight for one observation with innovation exactly e.

    With H = [1, x=1] and theta_pred = [0, 1], H @ theta_pred = 1, so y = 1 + e forces the
    innovation to be e. Calls the update directly (no predict side-effect on P).
    """
    from hrl.core.context import StepContext
    model = LinearGaussianSSM()
    theta_pred = np.array([0.0, 1.0])
    ctx = StepContext(t=0, y=1.0 + float(e), x=1.0, theta=theta_pred.copy(),
                      P=np.eye(2), R=r)
    GaussianUpdateStage(weight_fn=weight_fn).apply(ctx, model)
    return float(np.linalg.norm(ctx.theta - theta_pred)), ctx.w


def test_bounded_influence():
    """Vanilla influence grows ~linearly in |e|; WoLF is bounded and redescends."""
    van_small, _ = _state_shift(None, 5.0)
    van_big, _ = _state_shift(None, 50.0)
    assert van_big > 8.0 * van_small           # ~linear: 10x innovation -> ~10x state shift

    imq = IMQWeight(c=3.0)
    imq_peak, _ = _state_shift(imq, 200.0)
    imq_far, w_far = _state_shift(imq, 5000.0)
    assert imq_far < imq_peak                   # redescending past the influence peak
    van_far, _ = _state_shift(None, 5000.0)
    assert imq_far * 50.0 < van_far             # WoLF influence stays far below vanilla
    assert w_far < 0.01                          # extreme outlier is nearly ignored


def test_wolf_beats_vanilla_on_s3():
    """Monte Carlo: on heavy-tailed/contaminated S3, WoLF beta RMSE < vanilla KF RMSE."""
    n_paths, n = 40, 1500
    van, wolf = np.empty(n_paths), np.empty(n_paths)
    for i in range(n_paths):
        s = generate("S3", n_steps=n, seed=1000 + i)
        van[i] = metrics.beta_rmse(_pipe("none").run(s.y, s.x).beta, s.beta_true)
        wolf[i] = metrics.beta_rmse(_pipe("imq", c=3.0).run(s.y, s.x).beta, s.beta_true)
    win_rate = float(np.mean(wolf < van))
    assert wolf.mean() < van.mean()            # WoLF better on average
    assert win_rate >= 0.75                    # and on the large majority of paths
