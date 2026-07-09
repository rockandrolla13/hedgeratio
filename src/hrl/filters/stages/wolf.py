"""WoLF robust reweight stage (step 4) + weight-function policies.

IMQ is the primary method (Duran-Martin et al., ICML 2024, arXiv:2405.05646). Huber and
Student-t are references for comparison. Invariant: c -> inf recovers vanilla KF (w == 1).
"""
from __future__ import annotations

from ...core.context import StepContext
from ...core.protocols import StateSpaceModel, WeightFunction


class IMQWeight:
    """Inverse-multiquadric weight: w = (1 + e^2 / (c^2 S))^(-1/2)."""

    def __init__(self, c: float = 3.0) -> None:
        self.c = c

    def weight(self, e: float, S: float) -> float:
        # TODO: (1 + e**2 / (c**2 * S)) ** -0.5; c == inf -> 1.0 (assert in tests)
        raise NotImplementedError("IMQWeight.weight")


class HuberWeight:
    """Huberized innovation: unit weight inside k sigma, 1/|z| decay outside (k=1.345)."""

    def __init__(self, k: float = 1.345) -> None:
        self.k = k

    def weight(self, e: float, S: float) -> float:
        # TODO: z = e / sqrt(S); w = 1 if |z|<=k else k/|z|
        raise NotImplementedError("HuberWeight.weight")


class StudentTWeight:
    """Student-t measurement weight via a one-step scale-mixture (arXiv:1703.02428 style)."""

    def __init__(self, nu: float = 4.0) -> None:
        self.nu = nu

    def weight(self, e: float, S: float) -> float:
        # TODO: (nu + 1) / (nu + e^2 / S) normalised to (0, 1]
        raise NotImplementedError("StudentTWeight.weight")


class WolfReweightStage:
    """Step 4: sets ctx.w from a WeightFunction, after (e, S) and before the gain."""
    name = "wolf"

    def __init__(self, weight_fn: WeightFunction) -> None:
        self.weight_fn = weight_fn

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        # TODO: ctx.w = self.weight_fn.weight(ctx.e, ctx.S)
        raise NotImplementedError("WolfReweightStage")
