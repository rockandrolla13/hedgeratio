"""Placeholder T-tests blocked on concurrent phases (spec sections 2, 3).

These target Mechanism 2 (robust adaptive R_t, Phase 4 -- filters/stages/adaptive_r.py) and
Mechanism 3 (changepoint layer, Phase 6 -- filters/stages/changepoint.py), which are owned by
other agents and may not exist on this branch. Each is skipped but documents exactly what it will
assert, so the coverage is tracked and the tests can be fleshed out once the modules land.
"""
import pytest

pytestmark = pytest.mark.skip(reason="depends on Phase 4 (adaptive R) / Phase 6 (changepoint)")


def test_t7_single_outlier_impulse_bounded_r():
    """T7 (M2, deterministic): one 20-sigma innovation. Robust adaptive R inflates by at most
    ~1 + 2(1-lam)c^2, versus ~1 + 400(1-lam) for the naive (unweighted) EWMA vol recursion, and
    the gain 'deafness' lasts <= 3 days robust vs >= 20 naive."""


def test_t8_countercyclical_gain_s4():
    """T8 (M2, non-inferiority): on S4 stochastic vol, Var(delta beta | stress) is lower for the
    adaptive-R filter than fixed R (paired Wilcoxon), while calm-regime RMSE is not worse by more
    than 5% (one-sided non-inferiority, alpha=0.05)."""


def test_t9_vol_forecast_adequacy_mincer_zarnowitz():
    """T9 (M2, report): Mincer-Zarnowitz regression of e_t^2 on sigma2_t with HAC Wald yields a
    slope in [0.7, 1.3]."""


def test_t10_arl0_calibration():
    """T10 (M3, calibration): on 500 S1 null paths, tune the CUSUM threshold h so ARL0 ~ 500 days;
    binomial test on the realized false-alarm rate at alpha=0.05."""


def test_t11_masking_detection_power_s5():
    """T11 (M3, power, PRIMARY): on S5 with jump >= 0.15, detection within 60 days occurs on >= 90%
    of paths WITH WoLF weighting active; ROC traced as c varies."""


def test_t12_detector_outlier_immunity_s3():
    """T12 (M3, binomial): on contaminated S3 with no true breaks, the robust-weighted detector's
    false-alarm rate stays at or below the T10-calibrated rate; report the raw-z inflation."""


def test_t13_reset_vs_inflated_q_frontier_s5():
    """T13 (M3, inferential, PRIMARY): on S5, changepoint reset dominates Q-inflation x5/x10 on the
    (reconvergence tau, sd(delta beta)) frontier (paired Wilcoxon, Holm-Bonferroni)."""


def test_t14_hard_mode_s5_plus_s3():
    """T14 (M3, headline): on S5+S3 the full composite's beta RMSE is within 25% of the oracle KF
    told the true break and outlier locations."""
