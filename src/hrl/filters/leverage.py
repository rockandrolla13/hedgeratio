"""Leverage-point diagnostics and (diagnostic-gated) leverage-capped gain (Mechanism 5, spec 5).

The IMQ weight is a function of the innovation e_t only. An extreme regressor value x_t with an
ordinary innovation still produces an outsized state update through H_t = [1, x_t] -- a classic
leverage point that WoLF cannot see. Mechanism 5 measures this and, only if it demonstrably
bites (spec T20 gate), caps it.

    leverage stat   ell_t = H_t P_pred H_t' / median_250(H P H')
    cap (gated)     H_eff = H_t * min(1, sqrt(ell_max / ell_t)),  ell_max ~ 9,
                    applied to the GAIN only (the innovation still uses the true H_t).

`leverage_statistic` and `H_eff` are pure helpers for diagnostics. `LeverageCappedKalmanRW` is a
standalone filter (mirroring baselines.VanillaKalmanRW) that runs the cap in-line; it defaults to
`cap=False`, so out of the box it is exactly the (optionally WoLF-weighted) vanilla filter and the
cap is opt-in. It is intentionally NOT wired into Pipeline.from_config.
"""
from __future__ import annotations

import numpy as np

from ..core.results import FilterResult
from .stages.wolf import IMQWeight


def leverage_statistic(hph: np.ndarray, window: int = 250) -> np.ndarray:
    """Leverage path ell_t = hph_t / trailing_median(hph, `window`).

    `hph` is the H P_pred H' sequence (= S_t - R_t). The trailing median is strictly causal
    (uses hph over (t-window, t], i.e. up to and including t) and NaN until `window` samples
    exist, so ell is a diagnostic on the realized leverage relative to its recent typical level.
    """
    hph = np.asarray(hph, dtype=float)
    n = hph.shape[0]
    ell = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        med = np.median(hph[lo:t + 1])
        if t + 1 >= window and med > 0.0:
            ell[t] = hph[t] / med
    return ell


def hph_from_result(result: FilterResult, r: float) -> np.ndarray:
    """Recover the H P_pred H' path from a FilterResult under fixed measurement noise R = r."""
    return np.asarray(result.S, dtype=float) - float(r)


def H_eff(H: np.ndarray, ell: float, ell_max: float = 9.0) -> np.ndarray:
    """Leverage-capped observation row H_eff = H * min(1, sqrt(ell_max / ell)).

    ell <= ell_max (or non-finite) leaves H untouched; larger leverage shrinks the whole row so
    the effective leverage is clamped near ell_max. Used for the GAIN only.
    """
    H = np.asarray(H, dtype=float)
    if not np.isfinite(ell) or ell <= ell_max or ell <= 0.0:
        return H
    return H * float(np.sqrt(ell_max / ell))


class LeverageCappedKalmanRW:
    """Random-walk KF with an optional WoLF weight and an optional leverage cap on the gain.

    cap=False, weight_fn=None -> exactly VanillaKalmanRW.
    cap=False, weight_fn=IMQ  -> the WoLF composite (no cap): the T20 'damage' arm.
    cap=True,  weight_fn=IMQ  -> the leverage-capped composite: the T20 'repair' arm.

    The cap uses a strictly-causal trailing-median leverage statistic and applies H_eff to the
    gain / covariance update only; the innovation e_t always uses the true H_t.
    """

    def __init__(self, weight_fn=None, cap: bool = False, ell_max: float = 9.0,
                 window: int = 250, q_alpha: float = 1e-7, q_beta: float = 1e-5,
                 r: float = 1e-3, p0: float = 1.0) -> None:
        self.weight_fn = weight_fn
        self.cap = bool(cap)
        self.ell_max = float(ell_max)
        self.window = int(window)
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self.r = float(r)
        self.p0 = float(p0)

    @classmethod
    def wolf(cls, c: float = 3.0, cap: bool = False, **kw) -> "LeverageCappedKalmanRW":
        """Convenience: WoLF-IMQ(c) composite, capped or not."""
        return cls(weight_fn=IMQWeight(c=c), cap=cap, **kw)

    def run(self, y: np.ndarray, x: np.ndarray) -> FilterResult:
        """Filter (y, x); return FilterResult. `weights` holds w_t, `S` the nominal H P- H' + R."""
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
        w_arr = np.empty(n)

        hph_hist: list[float] = []
        for t in range(n):
            theta_pred = F @ theta
            P_pred = F @ P @ F.T + Q
            H = np.array([1.0, x[t]])
            hph = float(H @ P_pred @ H)
            e = y[t] - float(H @ theta_pred)
            S = hph + self.r                       # nominal predictive variance (true H)

            w = 1.0 if self.weight_fn is None else float(self.weight_fn.weight(e, S))

            # Leverage cap on the gain only (diagnostic-gated). Strictly-causal trailing median.
            H_gain = H
            if self.cap and len(hph_hist) >= self.window:
                med = float(np.median(hph_hist[-self.window:]))
                if med > 0.0:
                    H_gain = H_eff(H, hph / med, self.ell_max)

            hph_gain = float(H_gain @ P_pred @ H_gain)
            w2 = w * w
            K = (w2 * P_pred @ H_gain) / (w2 * hph_gain + self.r)
            theta = theta_pred + K * e
            P = (np.eye(2) - np.outer(K, H_gain)) @ P_pred

            alpha[t] = theta[0]
            beta[t] = theta[1]
            spread[t] = e
            S_arr[t] = S
            w_arr[t] = w
            hph_hist.append(hph)

        return FilterResult(beta=beta, alpha=alpha, spread=spread, S=S_arr,
                            weights=w_arr, resets=np.zeros(n, dtype=bool))
