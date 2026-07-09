"""AR(1)-anchored transition stage (alternative to plain predict, step 1).

Replaces the beta random walk with  beta_t = beta_bar_t + phi (beta_{t-1} - beta_bar_t) + w,
pulling beta toward a slow Johansen/TLS anchor beta_bar_t (causal, monthly, handoff-blended).
alpha stays a random walk. This stage and PredictStage are mutually exclusive.
"""
from __future__ import annotations

from ...core.context import StepContext
from ...core.protocols import AnchorProvider, StateSpaceModel


class AnchoredTransitionStage:
    """Step 1 (anchored): applies the AR(1) mean-reverting transition toward beta_bar_t."""
    name = "predict"   # occupies the single transition slot

    def __init__(self, anchor: AnchorProvider, phi: float = 0.99, q_beta: float = 1e-5) -> None:
        self.anchor = anchor
        self.phi = phi
        self.q_beta = q_beta

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        # TODO: beta_bar = self.anchor.anchor(ctx.t, y_hist, x_hist)
        #       theta-_beta = beta_bar + phi (theta_beta - beta_bar); alpha RW
        #       P- = F P F' + Q with Q_beta = q_beta (stationary var q_beta/(1-phi^2))
        raise NotImplementedError("AnchoredTransitionStage")
