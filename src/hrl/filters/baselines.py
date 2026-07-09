"""Baselines: rolling OLS, rolling TLS, and the vanilla random-walk Kalman filter.

These are the reference points against which the composite must be judged (design/plan Phase 1).
Rolling estimators are causal: beta_t uses only observations in (t-window, t].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.results import FilterResult


def steady_state_gain(q: float, r: float) -> float:
    """Steady-state Kalman gain for the scalar local-level (RW + noise) model.

    Solves the scalar Riccati fixed point P- = (q + sqrt(q^2 + 4 q r)) / 2 and returns
    K = P- / (P- + r). This is the EWRLS-equivalent gain; it ignores the x_t scaling of the
    beta observation row, so treat it as the calm-regime approximation for documentation.
    """
    p_pred = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    return float(p_pred / (p_pred + r))


def ewma_half_life(q: float, r: float) -> float:
    """Effective EWMA half-life (in steps) implied by q/r via the steady-state gain.

    The KF on a random-walk state is asymptotically an EWMA with forgetting lambda = 1 - K;
    the half-life is ln(0.5) / ln(1 - K). Returns inf when the gain vanishes (q -> 0).
    """
    k = steady_state_gain(q, r)
    if k <= 0.0 or k >= 1.0:
        return float("inf")
    return float(np.log(0.5) / np.log(1.0 - k))


def rolling_ols(y: np.ndarray, x: np.ndarray, window: int = 120) -> np.ndarray:
    """Trailing-window OLS hedge ratio beta_t (with intercept). Returns the beta path.

    Entries with fewer than `window` observations are NaN.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.shape[0]
    beta = np.full(n, np.nan)
    for t in range(window - 1, n):
        xs = x[t - window + 1: t + 1]
        ys = y[t - window + 1: t + 1]
        xm = xs.mean()
        ym = ys.mean()
        var = np.sum((xs - xm) ** 2)
        if var > 0.0:
            beta[t] = np.sum((xs - xm) * (ys - ym)) / var
    return beta


def rolling_tls(y: np.ndarray, x: np.ndarray, window: int = 120) -> np.ndarray:
    """Trailing-window total least squares (errors-in-variables) hedge ratio.

    Slope from the smallest right singular vector of the centered [x, y] window.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.shape[0]
    beta = np.full(n, np.nan)
    for t in range(window - 1, n):
        xs = x[t - window + 1: t + 1]
        ys = y[t - window + 1: t + 1]
        m = np.column_stack((xs - xs.mean(), ys - ys.mean()))
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        vx, vy = vt[-1]                 # direction of least variance
        if abs(vy) > 1e-12:
            beta[t] = -vx / vy
    return beta


class VanillaKalmanRW:
    """Random-walk state, fixed Q and R; equivalent to the Pipeline's vanilla path.

    On a static-beta Gaussian DGP, beta converges to truth and the steady-state gain matches
    the EWRLS equivalence (document the implied half-life). Provided as an independent
    reference implementation of the baseline KF.
    """

    def __init__(self, q_alpha: float = 1e-7, q_beta: float = 1e-5,
                 r: float = 1e-3, p0: float = 1.0) -> None:
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.r = r
        self.p0 = p0

    @classmethod
    def from_delta(cls, delta: float, r: float = 1e-3) -> "VanillaKalmanRW":
        """Chan/QuantStart parameterization: Q = delta/(1-delta) * I."""
        q = delta / (1.0 - delta)
        return cls(q_alpha=q, q_beta=q, r=r)

    def neg_log_likelihood(self, y: np.ndarray, x: np.ndarray, burn: int | None = None) -> float:
        """One-step-ahead prediction negative log-likelihood: -sum log N(e_t; 0, S_t)."""
        res = self.run(y, x)
        n = res.spread.shape[0]
        if burn is None:
            burn = max(20, n // 20)
        e = res.spread[burn:]
        s = res.S[burn:]
        return float(0.5 * np.sum(np.log(2.0 * np.pi * s) + e * e / s))

    def fit(self, y: np.ndarray, x: np.ndarray,
            bounds: tuple[float, float] = (1e-12, 1.0)) -> "VanillaKalmanRW":
        """Fit (q_alpha, q_beta, r) on a training split by maximising the prediction
        log-likelihood (L-BFGS-B in log-space). The caller passes the training slice; no
        data beyond it is touched (walk-forward discipline)."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        lo, hi = np.log(bounds[0]), np.log(bounds[1])

        def nll(log_params: np.ndarray) -> float:
            qa, qb, r = np.exp(log_params)
            return VanillaKalmanRW(qa, qb, r, self.p0).neg_log_likelihood(y, x)

        x0 = np.log([max(self.q_alpha, bounds[0]),
                     max(self.q_beta, bounds[0]),
                     max(self.r, bounds[0])])
        opt = minimize(nll, x0, method="L-BFGS-B", bounds=[(lo, hi)] * 3)
        self.q_alpha, self.q_beta, self.r = (float(v) for v in np.exp(opt.x))
        return self

    def run(self, y: np.ndarray, x: np.ndarray) -> FilterResult:
        """Run the plain KF (predict/update, w == 1) and return the FilterResult paths."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = y.shape[0]
        theta = np.zeros(2)
        P = np.eye(2) * self.p0
        F = np.eye(2)
        Q = np.diag([self.q_alpha, self.q_beta])

        beta = np.empty(n)
        alpha = np.empty(n)
        spread = np.empty(n)
        S_arr = np.empty(n)

        for t in range(n):
            theta = F @ theta
            P = F @ P @ F.T + Q
            H = np.array([1.0, x[t]])
            hph = float(H @ P @ H)
            e = y[t] - float(H @ theta)
            S = hph + self.r
            K = (P @ H) / S
            theta = theta + K * e
            P = (np.eye(2) - np.outer(K, H)) @ P
            alpha[t] = theta[0]
            beta[t] = theta[1]
            spread[t] = e
            S_arr[t] = S

        return FilterResult(beta=beta, alpha=alpha, spread=spread, S=S_arr,
                            weights=np.ones(n), resets=np.zeros(n, dtype=bool))
