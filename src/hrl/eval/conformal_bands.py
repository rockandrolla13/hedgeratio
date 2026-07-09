"""Adaptive conformal calibration of z-score trading bands (Mechanism 4, spec section 4).

Under heavy tails the nominal Gaussian band |z| <= 1.96 is badly miscalibrated (for t(3) the
95% point is |z| ~ 3.18). Adaptive Conformal Inference (Gibbs & Candes, 2021) fixes this online
without distributional assumptions:

    nonconformity   s_t = |z_t|,  z_t = e_t / sqrt(S_t)
    error indicator err_t = 1{ s_t > q_hat_t }
    online target   alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    quantile        q_hat_t = empirical (1 - alpha_t) quantile of the trailing W_cal scores.

The realized long-run miscoverage of ACI converges to the nominal alpha regardless of the score
distribution. Strictly causal: q_hat_t uses only scores strictly before t. Filter-agnostic --
consumes any FilterResult's `spread` (= innovation e_t) and `S` arrays.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _scores(spread: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Standardized-innovation nonconformity scores s_t = |e_t| / sqrt(S_t)."""
    spread = np.asarray(spread, dtype=float)
    S = np.asarray(S, dtype=float)
    return np.abs(spread) / np.sqrt(S)


def aci_bands(spread: np.ndarray, S: np.ndarray, alpha: float = 0.10,
              gamma: float = 0.01, W_cal: int = 250, warmup: int = 250) -> dict:
    """Adaptive Conformal Inference band path for standardized innovations.

    Parameters
    ----------
    spread, S : the FilterResult innovation and its nominal variance (z_t = spread/sqrt(S)).
    alpha     : target miscoverage (band nominal coverage = 1 - alpha).
    gamma     : ACI learning rate for the online alpha_t update.
    W_cal     : trailing calibration window length for the empirical quantile.
    warmup    : steps skipped before bands are evaluated (need a filled window).

    Returns a dict with, over the EVALUATED region (t >= warmup):
        q_hat     -- adaptive quantile path q_hat_t (nonconformity units, = a |z| threshold)
        alpha_t   -- online target-miscoverage path
        err       -- exceedance indicators err_t in {0, 1}
        eval_idx  -- integer time indices the above arrays correspond to
        coverage  -- realized coverage = 1 - mean(err)
        target    -- nominal coverage 1 - alpha
        n_eval    -- number of evaluated steps
    """
    s = _scores(spread, S)
    n = s.shape[0]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if warmup < 1 or W_cal < 1:
        raise ValueError("W_cal and warmup must be positive")

    q_path: list[float] = []
    a_path: list[float] = []
    err_path: list[int] = []
    idx: list[int] = []

    alpha_t = float(alpha)
    for t in range(n):
        lo = max(0, t - W_cal)
        window = s[lo:t]                       # strictly-causal trailing scores
        if t < warmup or window.size == 0:
            # still update alpha_t online so the state is warm once evaluation starts
            if window.size > 0:
                a_clip = min(max(alpha_t, 1e-6), 1.0 - 1e-6)
                q_hat = float(np.quantile(window, 1.0 - a_clip))
                err = int(s[t] > q_hat)
                alpha_t = alpha_t + gamma * (alpha - err)
            continue
        a_clip = min(max(alpha_t, 1e-6), 1.0 - 1e-6)
        q_hat = float(np.quantile(window, 1.0 - a_clip))
        err = int(s[t] > q_hat)
        q_path.append(q_hat)
        a_path.append(alpha_t)
        err_path.append(err)
        idx.append(t)
        alpha_t = alpha_t + gamma * (alpha - err)

    err_arr = np.asarray(err_path, dtype=int)
    n_eval = err_arr.shape[0]
    coverage = float(1.0 - err_arr.mean()) if n_eval else float("nan")
    return {
        "q_hat": np.asarray(q_path, dtype=float),
        "alpha_t": np.asarray(a_path, dtype=float),
        "err": err_arr,
        "eval_idx": np.asarray(idx, dtype=int),
        "coverage": coverage,
        "target": 1.0 - alpha,
        "n_eval": n_eval,
    }


def gaussian_bands(spread: np.ndarray, S: np.ndarray, alpha: float = 0.10,
                   warmup: int = 250) -> dict:
    """Fixed Gaussian-S band baseline: threshold q = Phi^{-1}(1 - alpha/2) on |z_t|.

    This is the naive calibration ACI is meant to beat; under heavy tails it systematically
    under-covers (mass-rejects the Kupiec test). Same return schema as `aci_bands` (with a
    constant q_hat and alpha_t path) for a like-for-like comparison.
    """
    s = _scores(spread, S)
    n = s.shape[0]
    q = float(stats.norm.ppf(1.0 - alpha / 2.0))
    idx = np.arange(warmup, n)
    err_arr = (s[warmup:] > q).astype(int)
    n_eval = err_arr.shape[0]
    coverage = float(1.0 - err_arr.mean()) if n_eval else float("nan")
    return {
        "q_hat": np.full(n_eval, q, dtype=float),
        "alpha_t": np.full(n_eval, alpha, dtype=float),
        "err": err_arr,
        "eval_idx": idx,
        "coverage": coverage,
        "target": 1.0 - alpha,
        "n_eval": n_eval,
    }
