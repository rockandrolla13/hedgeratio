"""Unit tests for the paired-inference toolkit (hrl.eval.stats)."""
import numpy as np
import pytest

from hrl.eval import stats


def test_paired_wilcoxon_detects_shift():
    """A uniform positive shift is flagged significant with a positive H-L estimate."""
    rng = np.random.default_rng(0)
    b = rng.normal(0.0, 1.0, 200)
    a = b + 0.5                                  # a strictly larger by 0.5
    stat, p, hl = stats.paired_wilcoxon(a, b)
    assert p < 1e-6
    assert abs(hl - 0.5) < 1e-9


def test_paired_wilcoxon_all_tied():
    """Identical inputs: no test possible, returns the degenerate (0, 1, 0)."""
    x = np.arange(10.0)
    assert stats.paired_wilcoxon(x, x) == (0.0, 1.0, 0.0)


def test_hodges_lehmann_matches_median_for_symmetric():
    """H-L of a symmetric sample equals its median."""
    d = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(stats.hodges_lehmann(d) - 0.0) < 1e-12


def test_holm_bonferroni_stepdown():
    """Holm rejects the smallest p only if it clears alpha/m, then steps down."""
    res = stats.holm_bonferroni([0.001, 0.04, 0.9], alpha=0.05)
    assert [r[1] for r in res] == [True, False, False]     # 0.04 > 0.05/2 -> stop
    # order is preserved even when input is unsorted
    res2 = stats.holm_bonferroni([0.9, 0.001], alpha=0.05)
    assert [r[1] for r in res2] == [False, True]


def test_holm_bonferroni_all_reject():
    """All-tiny p-values are all rejected."""
    res = stats.holm_bonferroni([1e-6, 2e-6, 3e-6], alpha=0.05)
    assert all(r[1] for r in res)


def test_relative_efficiency():
    """RE is the ratio of mean squared metrics."""
    r = np.array([2.0, 2.0])
    v = np.array([2.0, 2.0])
    assert stats.relative_efficiency(r, v) == pytest.approx(1.0)
    assert stats.relative_efficiency(np.array([1.0]), np.array([2.0])) == pytest.approx(0.25)


def test_kupiec_pof_correct_coverage_passes():
    """Observed failure rate equal to nominal gives LR ~ 0 (fail to reject)."""
    lr, p = stats.kupiec_pof(30, 300, 0.10)
    assert lr < 1e-9 and p > 0.99


def test_kupiec_pof_wrong_coverage_rejects():
    """Gross under-coverage (many exceedances) is rejected."""
    lr, p = stats.kupiec_pof(90, 300, 0.10)
    assert p < 1e-3


def test_christoffersen_independence_iid_vs_clustered():
    """Independence test passes iid exceedances and rejects clustered runs."""
    rng = np.random.default_rng(1)
    iid = (rng.random(3000) < 0.1).astype(int)
    _, p_iid = stats.christoffersen_independence(iid)
    clustered = np.zeros(3000, dtype=int)
    for k in range(0, 3000, 100):
        clustered[k:k + 10] = 1                  # regular bursts -> strong dependence
    _, p_cl = stats.christoffersen_independence(clustered)
    assert p_iid > 0.05
    assert p_cl < 0.01
