"""Mechanism 4 conformal-band tests (spec T15, T16).

T15 (unconditional coverage): under heavy tails the adaptive conformal (ACI) band's realized
coverage matches nominal and the Kupiec POF test fails to reject on the large majority of paths,
whereas the fixed Gaussian-S band is mass-rejected.
T16 (conditional coverage): ACI holds coverage near nominal across time-blocks far better than the
Gaussian band (block-wise coverage deviation, paired Wilcoxon).

Note on ACI + Christoffersen: the ACI feedback induces (mild) anti-correlation in the exceedance
sequence, so the Christoffersen *independence* test is not a fair pass/fail gate for ACI; T16 is
therefore posed as temporal (block-wise) conditional coverage, which ACI genuinely controls.
"""
import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import stats
from hrl.eval.conformal_bands import aci_bands, gaussian_bands
from hrl.filters.pipeline import Pipeline
from hrl.models.linear_ssm import LinearGaussianSSM

_R = 4e-4


def _vanilla(y, x):
    cfg = PipelineConfig(weight_fn="none", noise_model="fixed", r=_R, p0=10.0)
    return Pipeline.from_config(cfg, LinearGaussianSSM()).run(y, x)


def test_aci_causal_and_gaussian_shapes():
    """Both wrappers return aligned arrays over the warmed-up evaluation region."""
    s = generate("S3", n_steps=1200, seed=0)
    r = _vanilla(s.y, s.x)
    a = aci_bands(r.spread, r.S, alpha=0.10, gamma=0.01, W_cal=250, warmup=250)
    g = gaussian_bands(r.spread, r.S, alpha=0.10, warmup=250)
    for d in (a, g):
        assert d["q_hat"].shape == d["err"].shape == d["eval_idx"].shape
        assert d["n_eval"] == d["err"].shape[0] > 0
    assert a["eval_idx"][0] >= 250                         # warm-up respected (causal)
    assert np.all((a["err"] == 0) | (a["err"] == 1))


def test_aci_recovers_nominal_on_gaussian_null():
    """On clean Gaussian S1 the ACI coverage still tracks the nominal level."""
    s = generate("S1", n_steps=2500, seed=2)
    r = _vanilla(s.y, s.x)
    a = aci_bands(r.spread, r.S, alpha=0.10, gamma=0.01)
    assert abs(a["coverage"] - 0.90) < 0.05


def test_t15_unconditional_coverage_kupiec():
    """ACI passes the Kupiec POF test on ~all heavy-tailed paths; Gaussian-S is mass-rejected."""
    n_paths, n, alpha = 50, 2000, 0.10
    aci_ftr = 0
    gauss_ftr = 0
    aci_cov = np.empty(n_paths)
    for i in range(n_paths):
        s = generate("S3", n_steps=n, seed=11000 + i)
        r = _vanilla(s.y, s.x)
        a = aci_bands(r.spread, r.S, alpha=alpha, gamma=0.01)
        g = gaussian_bands(r.spread, r.S, alpha=alpha)
        aci_cov[i] = a["coverage"]
        _, pa = stats.kupiec_pof(int(a["err"].sum()), a["n_eval"], alpha)
        _, pg = stats.kupiec_pof(int(g["err"].sum()), g["n_eval"], alpha)
        aci_ftr += pa > 0.05
        gauss_ftr += pg > 0.05
    assert aci_ftr / n_paths >= 0.90                       # conformal: fail-to-reject (calibrated)
    assert gauss_ftr / n_paths <= 0.30                     # Gaussian-S: mass-rejected
    assert abs(aci_cov.mean() - (1 - alpha)) < 0.02        # coverage sits on nominal


def test_t16_conditional_coverage_blockwise():
    """ACI holds block-wise coverage near nominal better than Gaussian-S (temporal conditioning)."""
    n_paths, n, alpha, n_blocks = 60, 2000, 0.10, 8
    aci_dev = np.empty(n_paths)
    gauss_dev = np.empty(n_paths)

    def block_dev(err):
        covs = np.array([1.0 - b.mean() for b in np.array_split(err, n_blocks) if b.size])
        return float(np.mean(np.abs(covs - (1.0 - alpha))))

    for i in range(n_paths):
        s = generate("S3", n_steps=n, seed=11000 + i)
        r = _vanilla(s.y, s.x)
        aci_dev[i] = block_dev(aci_bands(r.spread, r.S, alpha=alpha, gamma=0.01)["err"])
        gauss_dev[i] = block_dev(gaussian_bands(r.spread, r.S, alpha=alpha)["err"])
    _, p, hl = stats.paired_wilcoxon(aci_dev, gauss_dev)
    assert np.median(aci_dev) < np.median(gauss_dev)
    assert hl < 0.0 and p < 0.01                            # ACI significantly closer to nominal
    assert float(np.mean(aci_dev < gauss_dev)) >= 0.70
