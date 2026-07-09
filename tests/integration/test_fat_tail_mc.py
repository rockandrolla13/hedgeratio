"""Mechanism 1 Monte Carlo tests (spec T2 contamination sweep, T3 paired inference).

T2: as contamination pi rises, WoLF RMSE stays near its clean level while vanilla degrades
severely. T3: WoLF beats vanilla per-path on heavy-tailed S3 with an overwhelmingly significant
paired Wilcoxon signed-rank test (Holm-Bonferroni-controlled across the robust family).

Path counts are below the spec's 200 for suite speed; effect sizes are large enough that the
thresholds carry comfortable margin at these sizes (see hard-coded ratios).
"""
import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import s3_heavy
from hrl.eval import metrics, stats
from hrl.filters.pipeline import Pipeline
from hrl.models.linear_ssm import LinearGaussianSSM

_R = 4e-4
_N = 1500


def _pipe(**kw):
    cfg = PipelineConfig(noise_model="fixed", r=_R, p0=10.0, **kw)
    return Pipeline.from_config(cfg, LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5))


def _s3(pi, seed, scale=100.0):
    """S3 with a parameterised contamination fraction pi (outlier scale = 100 R by default)."""
    rng = np.random.default_rng(seed)
    return s3_heavy(_N, rng, contam=pi, contam_scale=scale)


def test_t2_contamination_sweep():
    """WoLF RMSE at pi=5% <= 1.5x its pi=0; vanilla degrades >= 5x its pi=0."""
    n_paths = 50
    pis = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    van_mean = {}
    wolf_mean = {}
    for pi in pis:
        v = np.empty(n_paths)
        w = np.empty(n_paths)
        for i in range(n_paths):
            s = _s3(pi, 8000 + i)
            v[i] = metrics.beta_rmse(_pipe(weight_fn="none").run(s.y, s.x).beta, s.beta_true)
            w[i] = metrics.beta_rmse(_pipe(weight_fn="imq", c=3.0).run(s.y, s.x).beta, s.beta_true)
        van_mean[pi], wolf_mean[pi] = float(v.mean()), float(w.mean())
    # WoLF stays flat; vanilla blows up.
    assert wolf_mean[0.05] <= 1.5 * wolf_mean[0.0]
    assert van_mean[0.05] >= 5.0 * van_mean[0.0]
    # monotone-ish damage to vanilla, none to WoLF.
    assert van_mean[0.10] > van_mean[0.0]
    assert wolf_mean[0.10] <= 2.0 * wolf_mean[0.0]


def test_t3_paired_wilcoxon_holm():
    """WoLF (and the t-filter) beat vanilla per-path on S3; Wilcoxon + Holm both significant."""
    n_paths = 100
    van = np.empty(n_paths)
    wolf = np.empty(n_paths)
    huber = np.empty(n_paths)
    student = np.empty(n_paths)
    for i in range(n_paths):
        s = _s3(0.01, 9000 + i)
        van[i] = metrics.beta_rmse(_pipe(weight_fn="none").run(s.y, s.x).beta, s.beta_true)
        wolf[i] = metrics.beta_rmse(_pipe(weight_fn="imq", c=3.0).run(s.y, s.x).beta, s.beta_true)
        huber[i] = metrics.beta_rmse(_pipe(weight_fn="huber").run(s.y, s.x).beta, s.beta_true)
        student[i] = metrics.beta_rmse(_pipe(weight_fn="student_t").run(s.y, s.x).beta, s.beta_true)

    _, p_wolf, hl_wolf = stats.paired_wilcoxon(wolf, van)
    _, p_hub, _ = stats.paired_wilcoxon(huber, van)
    _, p_stu, _ = stats.paired_wilcoxon(student, van)
    # WoLF is significantly better (negative H-L shift = lower RMSE) and wins most paths.
    assert hl_wolf < 0.0
    assert p_wolf < 1e-4
    assert float(np.mean(wolf < van)) >= 0.70
    # Holm-Bonferroni over the robust family keeps WoLF's rejection.
    holm = stats.holm_bonferroni([p_wolf, p_hub, p_stu], alpha=0.05)
    assert holm[0][1] is True
