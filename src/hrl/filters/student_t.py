"""Student-t measurement filter (Mechanism 1.2 -- efficiency benchmark).

Scale-mixture representation of a heavy-tailed measurement:
    eps_t ~ t_nu(0, R)  <=>  eps_t | lambda_t ~ N(0, R / lambda_t),  lambda_t ~ Gamma(nu/2, nu/2).

Each step runs a short fixed-point (IRLS / EM) on the mixing weight lambda_t:
    lambda_hat = (nu + 1) / (nu + e_cur' R^-1 e_cur)
    R_eff      = R / lambda_hat
    KF update with R_eff, then recompute e_cur = y - H theta_cur.
Iterated 3-10 times to tol 1e-10 on lambda. This down-weights extreme innovations (bounded
influence) but, unlike WoLF-IMQ, does NOT redescend to zero -- lambda -> 0 as |e| -> inf but the
gain retains a floor, so it is the efficiency (not the pure breakdown) benchmark.

Mirrors filters.baselines.VanillaKalmanRW.run and returns the same FilterResult paths. Strictly
causal: theta_t uses only observations up to t. `weights` reports the converged lambda_t (the
per-step measurement responsibility), and S the effective predictive variance H P- H' + R_eff.
"""
from __future__ import annotations

import numpy as np

from ..core.results import FilterResult


class StudentTKalmanRW:
    """Random-walk state (alpha, beta) with a Student-t(nu) measurement, solved per step by
    the scale-mixture fixed-point. nu -> inf recovers the Gaussian (vanilla) filter."""

    def __init__(self, nu: float = 4.0, q_alpha: float = 1e-7, q_beta: float = 1e-5,
                 r: float = 1e-3, p0: float = 1.0,
                 max_iter: int = 10, tol: float = 1e-10) -> None:
        if nu <= 2.0:
            raise ValueError("nu must exceed 2 for a finite measurement variance")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.nu = float(nu)
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self.r = float(r)
        self.p0 = float(p0)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def run(self, y: np.ndarray, x: np.ndarray) -> FilterResult:
        """Filter (y, x) and return the per-step FilterResult paths."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if y.shape != x.shape:
            raise ValueError(f"y and x must align; got {y.shape} vs {x.shape}")
        n = y.shape[0]
        theta = np.zeros(2)
        P = np.eye(2) * self.p0
        F = np.eye(2)
        Q = np.diag([self.q_alpha, self.q_beta])

        beta = np.empty(n)
        alpha = np.empty(n)
        spread = np.empty(n)
        S_arr = np.empty(n)
        lam_arr = np.empty(n)

        for t in range(n):
            theta_pred = F @ theta
            P_pred = F @ P @ F.T + Q
            H = np.array([1.0, x[t]])
            hph = float(H @ P_pred @ H)
            e_pred = y[t] - float(H @ theta_pred)   # predictive innovation (fixed across iters)

            # Fixed point on the mixing weight lambda. e_cur is the residual w.r.t. the current
            # (lambda-dependent) state estimate; start it at the predictive residual.
            lam = 1.0
            e_cur = e_pred
            theta_new = theta_pred
            for _ in range(self.max_iter):
                lam_new = (self.nu + 1.0) / (self.nu + (e_cur * e_cur) / self.r)
                r_eff = self.r / lam_new
                S = hph + r_eff
                K = (P_pred @ H) / S
                theta_new = theta_pred + K * e_pred
                e_cur = y[t] - float(H @ theta_new)
                if abs(lam_new - lam) < self.tol:
                    lam = lam_new
                    break
                lam = lam_new

            r_eff = self.r / lam
            S = hph + r_eff
            K = (P_pred @ H) / S
            theta = theta_pred + K * e_pred
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            alpha[t] = theta[0]
            beta[t] = theta[1]
            spread[t] = e_pred
            S_arr[t] = S
            lam_arr[t] = lam

        return FilterResult(beta=beta, alpha=alpha, spread=spread, S=S_arr,
                            weights=lam_arr, resets=np.zeros(n, dtype=bool))
