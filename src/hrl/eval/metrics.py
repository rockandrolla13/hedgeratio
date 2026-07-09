"""Estimation, stability, spread-quality, and calibration metrics (design/plan Phase 8).

Estimation (vs ground truth): beta_rmse, beta_mae, max_beta_error, reconverge_time.
Stability: delta_beta_var (churn), turnover, sign_flips.
Spread quality: spread_adf, ou_half_life, pit_coverage (predictive-interval coverage).

Phase 1 implements the estimation + stability metrics; spread-quality/calibration metrics
arrive with the phases that need them.
"""
from __future__ import annotations

import numpy as np


def beta_rmse(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    """Root-mean-square hedge-ratio error."""
    beta_hat = np.asarray(beta_hat, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    return float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))


def beta_mae(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    """Mean absolute hedge-ratio error."""
    return float(np.mean(np.abs(np.asarray(beta_hat) - np.asarray(beta_true))))


def max_beta_error(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    """Maximum absolute hedge-ratio error."""
    return float(np.max(np.abs(np.asarray(beta_hat) - np.asarray(beta_true))))


def delta_beta_var(beta_hat: np.ndarray) -> float:
    """Realized variance of Delta beta_t (hedge-ratio churn)."""
    return float(np.var(np.diff(np.asarray(beta_hat, dtype=float))))


def turnover(beta_hat: np.ndarray, x: np.ndarray) -> float:
    """Rehedging turnover proxy: sum |Delta beta_t| * |x_t| (transaction-cost proxy)."""
    beta_hat = np.asarray(beta_hat, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(np.diff(beta_hat)) * np.abs(x[1:])))


def sign_flips(beta_hat: np.ndarray) -> int:
    """Number of sign changes in the Delta beta sequence (direction reversals)."""
    db = np.diff(np.asarray(beta_hat, dtype=float))
    s = np.sign(db)
    s = s[s != 0.0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def reconverge_time(beta_hat: np.ndarray, beta_true: np.ndarray,
                    break_times: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Steps to re-converge within tol of truth after each break (synthetic only)."""
    # TODO (Phase 6): per-break first-passage time back inside tol.
    raise NotImplementedError("reconverge_time")


def spread_adf(spread: np.ndarray) -> float:
    """ADF statistic of the (out-of-sample) filtered spread."""
    # TODO (Phase 5/8): statsmodels.tsa.stattools.adfuller.
    raise NotImplementedError("spread_adf")


def ou_half_life(spread: np.ndarray) -> float:
    """Estimated OU mean-reversion half-life of the filtered spread."""
    # TODO (Phase 5/8).
    raise NotImplementedError("ou_half_life")


def pit_coverage(spread: np.ndarray, S: np.ndarray, nominal: float = 0.90) -> float:
    """Empirical coverage of nominal 1-step predictive intervals (PIT calibration)."""
    # TODO (Phase 8): fraction of |spread| within z_nominal * sqrt(S).
    raise NotImplementedError("pit_coverage")
