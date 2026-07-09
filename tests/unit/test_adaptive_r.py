"""Phase 4 tests for the adaptive measurement-noise stage (EWMA / GARCH / VB-AKF)."""
import numpy as np
import pytest

from hrl.config import PipelineConfig
from hrl.core.context import StepContext
from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.pipeline import Pipeline
from hrl.filters.stages.adaptive_r import (
    AdaptiveRStage,
    EWMANoise,
    GARCHNoise,
    VBAKFNoise,
    make_noise_model,
)
from hrl.filters.stages.wolf import IMQWeight
from hrl.models.linear_ssm import LinearGaussianSSM


def _pipe(noise_model, weight_fn="none", r=4e-4, lam=0.94, r_floor=1e-8, q_beta=1e-5):
    model = LinearGaussianSSM(q_alpha=1e-7, q_beta=q_beta)
    cfg = PipelineConfig(weight_fn=weight_fn, noise_model=noise_model, r=r, p0=10.0,
                         lam=lam, r_floor=r_floor)
    return Pipeline.from_config(cfg, model)


def _drive_ewma(noise, prev_we_seq, r0):
    """Feed a sequence of lagged weighted innovations through a noise model; return R_t path."""
    ctx = StepContext(t=0, y=0.0, x=0.0, theta=np.zeros(2), P=np.eye(2), R=r0)
    out = []
    for pwe in prev_we_seq:
        ctx.R = r0                       # mimics the per-step reset in Pipeline.run
        if pwe is None:
            ctx.extra.pop("prev_we", None)
        else:
            ctx.extra["prev_we"] = pwe
        out.append(noise.update(ctx))
    return np.array(out)


# --- smoke ----------------------------------------------------------------------------------

def test_import_and_instantiate():
    """Smoke: noise models, stage, and factory construct."""
    assert EWMANoise(lam=0.94).lam == 0.94
    assert GARCHNoise() is not None
    assert VBAKFNoise().n_iter == 3
    assert AdaptiveRStage(EWMANoise()).name == "adaptive_r"


def test_factory_maps_kinds():
    """make_noise_model mirrors make_weight_fn: 'fixed' -> None, others -> the model."""
    cfg = PipelineConfig(lam=0.97, r_floor=1e-9)
    assert make_noise_model("fixed", cfg) is None
    ewma = make_noise_model("ewma", cfg)
    assert isinstance(ewma, EWMANoise) and ewma.lam == 0.97 and ewma.r_floor == 1e-9
    assert isinstance(make_noise_model("garch", cfg), GARCHNoise)
    assert isinstance(make_noise_model("vbakf", cfg), VBAKFNoise)
    with pytest.raises(ValueError):
        make_noise_model("nope", cfg)


def test_ewma_floors_and_lags():
    """R_t honours the floor and uses the *lagged* innovation (first step == seed R)."""
    n = EWMANoise(lam=0.9, r_floor=1e-3)
    # first step: no prev_we -> seed from ctx.R; floored below r_floor
    out = _drive_ewma(n, [None, 0.0, 0.0], r0=1e-6)
    assert np.all(out >= 1e-3 - 1e-15)          # floor active (seed 1e-6 < floor)
    # a big prev_we only affects the *next* R, not the current one
    n2 = EWMANoise(lam=0.9, r_floor=1e-12)
    out2 = _drive_ewma(n2, [None, None, 1.0], r0=1e-4)
    assert out2[1] == pytest.approx(1e-4)       # still the seed (prev_we was None)
    assert out2[2] > out2[1]                    # the 1.0 innovation lifts R next step


# --- saturation bound (deterministic) -------------------------------------------------------

def test_saturation_bound_with_wolf():
    """WoLF caps |w e| <= c sqrt(S), so one 20-sigma print inflates the EWMA vol state by at
    most ~(1-lam) c^2 relative to R; naive (no-weight) EWMA inflates far more."""
    lam, c, r_true = 0.94, 3.0, 1e-3
    S = r_true                                  # take hph negligible so S ~ R
    imq = IMQWeight(c=c)
    e_out = 20.0 * np.sqrt(S)                    # a 20-sigma innovation
    we_wolf = imq.weight(e_out, S) * e_out       # weighted -> saturated near c sqrt(S)
    assert abs(we_wolf) <= c * np.sqrt(S) + 1e-12

    calm = np.sqrt(r_true)                       # constant calm draw -> EWMA fixed point == r_true
    warm = [None] + [calm] * 20
    R_wolf = _drive_ewma(EWMANoise(lam=lam, r_floor=1e-15), warm + [we_wolf] + [calm] * 20, r_true)
    R_naive = _drive_ewma(EWMANoise(lam=lam, r_floor=1e-15), warm + [e_out] + [calm] * 20, r_true)

    bound = 1.0 + 2.0 * (1.0 - lam) * c * c      # ~2.08
    assert R_wolf.max() / r_true <= bound        # WoLF keeps the inflation bounded
    assert R_naive.max() / r_true > 5.0 * (R_wolf.max() / r_true)   # naive blows up (~17x here)


# --- counter-cyclical gain on S4 ------------------------------------------------------------

def test_ewma_counter_cyclical_on_s4():
    """Paired MC on S4: EWMA R shrinks Var(db) in stress vs fixed R, without degrading calm
    tracking (counter-cyclical gain). Uses the S4 regime labels."""
    n_paths, n, R, lam = 40, 2000, 4e-4, 0.94
    v_fix = np.empty(n_paths)
    v_ew = np.empty(n_paths)
    calm_fix = np.empty(n_paths)
    calm_ew = np.empty(n_paths)
    for i in range(n_paths):
        s = generate("S4", n_steps=n, seed=500 + i)
        stress = s.regime[1:].astype(bool)       # aligned to np.diff(beta)
        calm = s.regime == 0
        bf = _pipe("fixed", r=R).run(s.y, s.x).beta
        be = _pipe("ewma", r=R, lam=lam).run(s.y, s.x).beta
        v_fix[i] = np.var(np.diff(bf)[stress])
        v_ew[i] = np.var(np.diff(be)[stress])
        calm_fix[i] = metrics.beta_rmse(bf[calm], s.beta_true[calm])
        calm_ew[i] = metrics.beta_rmse(be[calm], s.beta_true[calm])

    assert v_ew.mean() < v_fix.mean()                    # stress churn falls on average
    assert float(np.mean(v_ew < v_fix)) >= 0.85          # ... and on the large majority of paths
    assert calm_ew.mean() <= 1.05 * calm_fix.mean()      # calm tracking non-inferior (~5%)


# --- VB-AKF fixed point ---------------------------------------------------------------------

def test_vbakf_fixed_point_converges():
    """The VB-AKF (state, scale) fixed point is Cauchy and converges well within n_iter."""
    vb = VBAKFNoise(rho=0.98, n_iter=6, r_floor=1e-12)
    ctx = StepContext(t=0, y=5.0, x=float(generate("S4", 5, seed=1).x[0]),
                      theta=np.array([0.0, 1.0]), P=np.eye(2) * 10.0, R=4e-4)
    R = vb.update(ctx)
    hist = ctx.extra["vbakf_R_iters"]
    assert len(hist) == 6 and R > 0.0
    deltas = np.abs(np.diff(hist))
    assert np.all(deltas[1:] <= deltas[:-1] + 1e-18)     # monotone-shrinking increments
    assert deltas[-1] / hist[-1] < 1e-3                  # converged (relative step tiny)


def test_vbakf_runs_over_a_path():
    """VB-AKF composes into the pipeline and produces a finite, positive R path on S4."""
    s = generate("S4", n_steps=800, seed=3)
    res = _pipe("vbakf", r=4e-4).run(s.y, s.x)
    assert np.all(np.isfinite(res.beta)) and np.all(res.S > 0.0)


# --- pipeline composition -------------------------------------------------------------------

def test_pipeline_inserts_adaptive_stage_in_order():
    """noise_model != 'fixed' inserts AdaptiveRStage at position 2 (predict -> R -> update)."""
    pipe = _pipe("ewma", weight_fn="imq")
    names = [getattr(s, "name", None) for s in pipe.stages]
    assert names == ["predict", "adaptive_r", "update"]
    # 'fixed' keeps the two-stage pipeline
    assert [getattr(s, "name", None) for s in _pipe("fixed").stages] == ["predict", "update"]


def test_ewma_composes_with_wolf():
    """EWMA + WoLF (imq) is a legal composition and runs (the intended stress-robust stack)."""
    s = generate("S4", n_steps=600, seed=7)
    res = _pipe("ewma", weight_fn="imq").run(s.y, s.x)
    assert np.all(np.isfinite(res.beta))
    assert np.all((res.weights > 0.0) & (res.weights <= 1.0 + 1e-12))


def test_vbakf_wolf_mutually_exclusive():
    """VB-AKF and WoLF must not co-iterate: the pipeline rejects the composition."""
    model = LinearGaussianSSM()
    cfg = PipelineConfig(weight_fn="imq", noise_model="vbakf")
    with pytest.raises(ValueError, match="co-iterate"):
        Pipeline.from_config(cfg, model)
    # but VB-AKF with no weighting is fine
    assert Pipeline.from_config(PipelineConfig(weight_fn="none", noise_model="vbakf"), model)
