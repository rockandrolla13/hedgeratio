"""Adaptive measurement-noise stage (step 2) + R_t policies.

Primary: EWMA plug-in on robustly-weighted squared innovations (lagged => no within-step
circularity). Secondary: GARCH(1,1) recursion on the same lagged innovation, and VB-AKF
(inverse-gamma conjugate, fixed-point per step; Sarkka-Nummenmaa 2009). VB-AKF is mutually
exclusive with WoLF within a step -- the Pipeline enforces this.

All three read the *previous* step's (weighted) innovation from ``ctx.extra["prev_we"]``,
stashed by GaussianUpdateStage, so R_t is a function of information available before the
current innovation is formed. Per-path scratch (the vol / IG state) lives in ``ctx.extra``,
keeping the stages free of cross-path mutable state.
"""
from __future__ import annotations

import numpy as np

from ...core.context import StepContext
from ...core.protocols import MeasurementNoiseModel, StateSpaceModel


class EWMANoise:
    """EWMA plug-in: sigma2_t = lam sigma2_{t-1} + (1-lam) (w_{t-1} e_{t-1})^2, floored.

    Uses the *lagged* weighted innovation, so there is no within-step circularity: R_t is
    fixed before the current innovation exists. sigma2 is initialised from ctx.R on the first
    step and carried in ctx.extra. With WoLF upstream, |w e| <= c sqrt(S) saturates, so one
    outlier can inflate sigma2 by at most (1-lam) c^2 S -- the counter-cyclical bound.
    """

    def __init__(self, lam: float = 0.94, r_floor: float = 1e-8) -> None:
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must lie in (0, 1)")
        self.lam = float(lam)
        self.r_floor = float(r_floor)

    def update(self, ctx: StepContext) -> float:
        sigma2 = ctx.extra.get("ewma_sigma2")
        if sigma2 is None:                       # first step: seed from the prior R
            sigma2 = float(ctx.R)
        prev_we = ctx.extra.get("prev_we")
        if prev_we is not None:                  # lagged weighted innovation
            sigma2 = self.lam * sigma2 + (1.0 - self.lam) * (prev_we * prev_we)
        ctx.extra["ewma_sigma2"] = sigma2
        return max(sigma2, self.r_floor)


class GARCHNoise:
    """GARCH(1,1) variance recursion: sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}.

    Parameters are constructor arguments (fitting on a training split is out of scope). Like
    EWMANoise it consumes the lagged (weighted) innovation and floors the result.
    """

    def __init__(self, omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.94,
                 r_floor: float = 1e-8) -> None:
        self.omega, self.alpha, self.beta = float(omega), float(alpha), float(beta)
        self.r_floor = float(r_floor)

    def update(self, ctx: StepContext) -> float:
        sigma2 = ctx.extra.get("garch_sigma2")
        if sigma2 is None:
            sigma2 = float(ctx.R)
        prev_we = ctx.extra.get("prev_we")
        if prev_we is not None:
            sigma2 = self.omega + self.alpha * (prev_we * prev_we) + self.beta * sigma2
        ctx.extra["garch_sigma2"] = sigma2
        return max(sigma2, self.r_floor)


class VBAKFNoise:
    """Variational-Bayes adaptive R (inverse-gamma) with forgetting rho; 2-4 fixed-point iters.

    Sarkka & Nummenmaa (2009). The measurement variance is modelled R ~ Inv-Gamma(a, b) with
    E[1/R] = a/b, so the filter's effective noise is R_hat = b/a. Each step: forget the IG
    params (a-, b- = rho a, rho b), then iterate a coupled fixed point over (state, b) with a
    held at a- + 1/2. This produces the converged R_t that the downstream GaussianUpdateStage
    then uses -- it does NOT itself commit the state update. Reconstructs H = [1, x] from the
    linear-Gaussian observation model. Reference column; not composed with WoLF.
    """

    def __init__(self, rho: float = 0.98, n_iter: int = 3, r_floor: float = 1e-8,
                 a0: float = 2.0) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must lie in (0, 1]")
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        self.rho, self.n_iter, self.r_floor = float(rho), int(n_iter), float(r_floor)
        self.a0 = float(a0)

    def fixed_point(self, m_pred: np.ndarray, P_pred: np.ndarray, H: np.ndarray,
                    y: float, a: float, b_minus: float) -> tuple[float, list[float]]:
        """Iterate the coupled (state, scale) VB fixed point; return (R_hat, per-iter R history).

        Pure/side-effect-free helper so a test can assert Cauchy convergence within n_iter.
        """
        hph_pred = float((H @ P_pred @ H.T)[0, 0])
        y_pred = float((H @ m_pred)[0])
        b = b_minus
        history: list[float] = []
        for _ in range(self.n_iter):
            R = b / a
            S = hph_pred + R
            k = (P_pred @ H.T).ravel() / S                 # Kalman gain (d,)
            m = m_pred + k * (y - y_pred)
            P = P_pred - S * np.outer(k, k)
            resid = y - float((H @ m)[0])
            b = b_minus + 0.5 * (resid * resid + float((H @ P @ H.T)[0, 0]))
            history.append(b / a)
        return b / a, history

    def update(self, ctx: StepContext) -> float:
        a_prev = ctx.extra.get("vbakf_a")
        b_prev = ctx.extra.get("vbakf_b")
        if a_prev is None:                       # first step: seed so b/a == prior R
            a_prev = self.a0
            b_prev = float(ctx.R) * a_prev
        a_minus = self.rho * a_prev
        b_minus = self.rho * b_prev
        a = a_minus + 0.5                        # fixed once the state dimension is known
        H = np.array([[1.0, ctx.x]])
        _, history = self.fixed_point(ctx.theta, ctx.P, H, ctx.y, a, b_minus)
        b = a * history[-1]
        ctx.extra["vbakf_a"] = a
        ctx.extra["vbakf_b"] = b
        ctx.extra["vbakf_R_iters"] = history
        return max(b / a, self.r_floor)


class AdaptiveRStage:
    """Step 2: sets ctx.R from a MeasurementNoiseModel before the innovation is formed."""
    name = "adaptive_r"

    def __init__(self, noise_model: MeasurementNoiseModel) -> None:
        self.noise_model = noise_model

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        ctx.R = self.noise_model.update(ctx)


def make_noise_model(kind: str, cfg) -> MeasurementNoiseModel | None:
    """Build a MeasurementNoiseModel from a config string ('fixed' -> None)."""
    if kind == "fixed":
        return None
    if kind == "ewma":
        return EWMANoise(lam=cfg.lam, r_floor=cfg.r_floor)
    if kind == "garch":
        return GARCHNoise(r_floor=cfg.r_floor)
    if kind == "vbakf":
        return VBAKFNoise(r_floor=cfg.r_floor)
    raise ValueError(f"unknown noise_model: {kind!r}")
