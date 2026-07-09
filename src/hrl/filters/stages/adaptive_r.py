"""Adaptive measurement-noise stage (step 2) + R_t policies.

Primary: EWMA plug-in on robustly-weighted squared innovations (lagged => no within-step
circularity). Secondary: GARCH(1,1) fit on the training split, and VB-AKF (inverse-gamma
conjugate, fixed-point per step; Sarkka-Nummenmaa 2009). VB-AKF is mutually exclusive with
WoLF within a step -- the Pipeline enforces this.
"""
from __future__ import annotations

from ...core.context import StepContext
from ...core.protocols import MeasurementNoiseModel, StateSpaceModel


class EWMANoise:
    """R_t = lam R_{t-1} + (1-lam) (w_{t-1} e_{t-1})^2 . Uses lagged weighted innovation."""

    def __init__(self, lam: float = 0.94, r_floor: float = 1e-8) -> None:
        self.lam = lam
        self.r_floor = r_floor

    def update(self, ctx: StepContext) -> float:
        # TODO: read prior weighted innovation from ctx.extra; EWMA; clamp at r_floor
        raise NotImplementedError("EWMANoise.update")


class GARCHNoise:
    """GARCH(1,1) innovation-variance forecast, parameters fit on the training split."""

    def __init__(self, omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.94) -> None:
        self.omega, self.alpha, self.beta = omega, alpha, beta

    def update(self, ctx: StepContext) -> float:
        # TODO: sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}
        raise NotImplementedError("GARCHNoise.update")


class VBAKFNoise:
    """Variational-Bayes adaptive R (inverse-gamma) with forgetting factor rho; 2-4 iters."""

    def __init__(self, rho: float = 0.98, n_iter: int = 3, r_floor: float = 1e-8) -> None:
        self.rho, self.n_iter, self.r_floor = rho, n_iter, r_floor

    def update(self, ctx: StepContext) -> float:
        # TODO: fixed-point on inverse-gamma posterior; assert convergence in tests
        raise NotImplementedError("VBAKFNoise.update")


class AdaptiveRStage:
    """Step 2: sets ctx.R from a MeasurementNoiseModel before the innovation is formed."""
    name = "adaptive_r"

    def __init__(self, noise_model: MeasurementNoiseModel) -> None:
        self.noise_model = noise_model

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        # TODO: ctx.R = self.noise_model.update(ctx)
        raise NotImplementedError("AdaptiveRStage")
