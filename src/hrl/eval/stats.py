"""Paired-inference toolkit for the fat-tail robustness test suite (spec section 6).

Small, pure, typed helpers used across the T-tests:
  * paired_wilcoxon      -- Wilcoxon signed-rank + Hodges-Lehmann shift (T3, T8, T13, T20).
  * holm_bonferroni      -- step-down FWER control at a family alpha (spec section 6).
  * relative_efficiency  -- mean(rmse_robust^2) / mean(rmse_vanilla^2) (T4).
  * kupiec_pof           -- Kupiec (1995) proportion-of-failures LR coverage test (T15).
  * christoffersen_independence / christoffersen_cc -- (T16) independence + joint LR.

Everything is strictly deterministic given its inputs; no RNG, no I/O.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def hodges_lehmann(d: np.ndarray) -> float:
    """One-sample Hodges-Lehmann location estimate: median of Walsh averages of `d`.

    For paired data pass the per-pair differences d = a - b; the result is the robust
    median shift (positive => `a` typically larger than `b`).
    """
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    if n == 0:
        return float("nan")
    i, j = np.triu_indices(n, k=0)          # i <= j: includes the diagonal (self-averages)
    walsh = 0.5 * (d[i] + d[j])
    return float(np.median(walsh))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Wilcoxon signed-rank test on paired samples `a` vs `b`.

    Returns (statistic, p_value, hodges_lehmann_shift) where the shift is the H-L median
    of the pairwise differences d = a - b. p is the two-sided p-value. When every pair is
    tied (d == 0) scipy cannot run; we return (0.0, 1.0, 0.0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must align; got {a.shape} vs {b.shape}")
    d = a - b
    hl = hodges_lehmann(d)
    if np.allclose(d, 0.0):
        return 0.0, 1.0, hl
    res = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
    return float(res.statistic), float(res.pvalue), float(hl)


def holm_bonferroni(pvalues, alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down procedure controlling the FWER at `alpha`.

    Returns a list of (p_value, reject) in the ORIGINAL input order. A hypothesis is
    rejected iff, at its sorted rank k (0-based, ascending p), p_(k) <= alpha / (m - k)
    AND all lower-ranked hypotheses were also rejected (step-down monotonicity).
    """
    p = np.asarray(list(pvalues), dtype=float)
    m = p.shape[0]
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    reject = np.zeros(m, dtype=bool)
    still = True
    for k, idx in enumerate(order):
        thresh = alpha / (m - k)
        if still and p[idx] <= thresh:
            reject[idx] = True
        else:
            still = False
    return [(float(p[i]), bool(reject[i])) for i in range(m)]


def relative_efficiency(rmse_robust, rmse_vanilla) -> float:
    """Relative efficiency mean(rmse_robust^2) / mean(rmse_vanilla^2) (spec T4).

    RE <= 1 means the robust estimator loses no efficiency under the (clean) null; RE > 1
    quantifies the variance price paid for robustness.
    """
    r = np.asarray(rmse_robust, dtype=float)
    v = np.asarray(rmse_vanilla, dtype=float)
    denom = float(np.mean(v * v))
    if denom == 0.0:
        return float("inf")
    return float(np.mean(r * r) / denom)


def kupiec_pof(failures: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures unconditional-coverage LR test.

    `failures` exceedances out of `n` observations against expected failure prob `p`
    (= nominal miscoverage, e.g. 0.10 for a 90% band). Returns (LR_pof, p_value) with the
    statistic ~ chi^2(1) under H0. Fail-to-reject => coverage matches nominal.
    """
    x, N = int(failures), int(n)
    if N <= 0:
        return float("nan"), float("nan")
    pi = x / N
    # Null log-likelihood (Bernoulli at rate p).
    ll0 = x * np.log(p) + (N - x) * np.log(1.0 - p)
    # Unrestricted MLE log-likelihood; handle the pi in {0, 1} boundary limits (0*log0 -> 0).
    ll1 = 0.0
    if x > 0:
        ll1 += x * np.log(pi)
    if x < N:
        ll1 += (N - x) * np.log(1.0 - pi)
    lr = float(-2.0 * (ll0 - ll1))
    lr = max(lr, 0.0)
    pval = float(stats.chi2.sf(lr, df=1))
    return lr, pval


def christoffersen_independence(hits) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test on a 0/1 exceedance sequence.

    Tests whether exceedances cluster (Markov dependence) versus arrive independently.
    Returns (LR_ind, p_value), statistic ~ chi^2(1). Degenerate transition counts
    (no state ever visited) return LR = 0, p = 1.
    """
    h = np.asarray(hits, dtype=int)
    if h.shape[0] < 2:
        return 0.0, 1.0
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return 0.0, 1.0
    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / (n0 + n1)
    if pi in (0.0, 1.0):
        return 0.0, 1.0

    def _xlogy(k: int, q: float) -> float:
        return k * np.log(q) if k > 0 else 0.0

    ll_null = _xlogy(n01 + n11, pi) + _xlogy(n00 + n10, 1.0 - pi)
    ll_alt = (_xlogy(n01, pi01) + _xlogy(n00, 1.0 - pi01)
              + _xlogy(n11, pi11) + _xlogy(n10, 1.0 - pi11))
    lr = max(float(-2.0 * (ll_null - ll_alt)), 0.0)
    return lr, float(stats.chi2.sf(lr, df=1))


def christoffersen_cc(hits, p: float) -> tuple[float, float]:
    """Christoffersen conditional-coverage joint LR = Kupiec POF + independence.

    Statistic ~ chi^2(2). Fail-to-reject => correct unconditional coverage AND independent
    exceedances (spec T16).
    """
    h = np.asarray(hits, dtype=int)
    lr_pof, _ = kupiec_pof(int(h.sum()), h.shape[0], p)
    lr_ind, _ = christoffersen_independence(h)
    lr_cc = float(lr_pof + lr_ind)
    return lr_cc, float(stats.chi2.sf(lr_cc, df=2))
