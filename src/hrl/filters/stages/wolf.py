"""WoLF robust reweight policies (step 4), injected into GaussianUpdateStage.

IMQ is the primary method (Duran-Martin et al., ICML 2024, arXiv:2405.05646). Huber and
Student-t are references for comparison, expressed here as weight functions feeding the same
WoLF w^2 gain (a generalised-Bayes family); the "pure" Huber innovation-clip and the exact
scale-mixture t-filter are separate mechanisms deferred as refinements.

Invariant: c -> inf recovers the vanilla Kalman filter (w == 1). All weights lie in (0, 1].
The weight is a function of the innovation e and its nominal variance S = H P- H' + R only.
"""
from __future__ import annotations

import numpy as np

from ...core.context import StepContext
from ...core.protocols import StateSpaceModel, WeightFunction


class IMQWeight:
    """Inverse-multiquadric weight: w = (1 + e^2 / (c^2 S))^(-1/2).

    Redescending influence: as |e| -> inf the effective state update -> 0, so a single huge
    outlier cannot move the state. c -> inf gives w == 1 (vanilla KF).
    """

    def __init__(self, c: float = 3.0) -> None:
        if c <= 0.0:
            raise ValueError("c must be positive")
        self.c = float(c)

    def weight(self, e: float, S: float) -> float:
        return float((1.0 + (e * e) / (self.c * self.c * S)) ** -0.5)


class HuberWeight:
    """Huberized weight on the standardized innovation z = e / sqrt(S): w = min(1, k/|z|)."""

    def __init__(self, k: float = 1.345) -> None:
        self.k = float(k)

    def weight(self, e: float, S: float) -> float:
        z = abs(e) / np.sqrt(S)
        if z <= self.k:
            return 1.0
        return float(self.k / z)


class StudentTWeight:
    """Student-t IRLS weight w = sqrt((nu+1)/(nu+z^2)), capped at 1 to stay in (0, 1]."""

    def __init__(self, nu: float = 4.0) -> None:
        self.nu = float(nu)

    def weight(self, e: float, S: float) -> float:
        z2 = (e * e) / S
        w2 = (self.nu + 1.0) / (self.nu + z2)
        return float(np.sqrt(min(1.0, w2)))


class WolfReweightStage:
    """Deprecated separate-stage form. Superseded by injecting a WeightFunction into
    GaussianUpdateStage, because w_t depends on (e_t, S_t) computed inside the update.

    Retained only for API compatibility; not used by Pipeline.from_config.
    """
    name = "wolf"

    def __init__(self, weight_fn: WeightFunction) -> None:
        self.weight_fn = weight_fn

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        ctx.w = float(self.weight_fn.weight(ctx.e, ctx.S))


def make_weight_fn(kind: str, cfg) -> WeightFunction | None:
    """Build a WeightFunction from a config string ('none' -> None)."""
    if kind == "none":
        return None
    if kind == "imq":
        return IMQWeight(c=cfg.c)
    if kind == "huber":
        return HuberWeight()
    if kind == "student_t":
        return StudentTWeight()
    raise ValueError(f"unknown weight_fn: {kind!r}")
