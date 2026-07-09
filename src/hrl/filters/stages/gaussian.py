"""Baseline Gaussian stages: predict (step 1) and update (steps 3-5).

GaussianUpdateStage computes the innovation (e, S), sets the robust weight w from an injected
WeightFunction (or w = 1 for vanilla), and applies the (weighted) gain. Injecting the weight
here -- rather than as a separate ordered stage -- is deliberate: w_t depends on (e_t, S_t),
which do not exist until the update begins, so weighting cannot precede the innovation. When
weight_fn is None and R is fixed, this reduces exactly to the vanilla Kalman update.
"""
from __future__ import annotations

import numpy as np

from ...core.context import StepContext
from ...core.protocols import StateSpaceModel, WeightFunction


class PredictStage:
    """Step 1: theta- = F theta, P- = F P F' + Q using the model's transition."""
    name = "predict"

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        F, Q = model.transition(ctx)
        ctx.theta = F @ ctx.theta
        ctx.P = F @ ctx.P @ F.T + Q


class GaussianUpdateStage:
    """Steps 3-5: innovation (e, S), robust weight w, and weighted-gain update.

    WoLF gain: K = w^2 P- H' / (w^2 H P- H' + R). w == 1 recovers the vanilla KF.
    The reported S = H P- H' + R is the *nominal* predictive variance (used for z-scores /
    PIT), independent of the down-weighting applied to the gain.
    """
    name = "update"

    def __init__(self, weight_fn: WeightFunction | None = None) -> None:
        self.weight_fn = weight_fn

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        H = model.observation(ctx)              # (1, d)
        theta_pred = ctx.theta
        P_pred = ctx.P

        hph = float((H @ P_pred @ H.T)[0, 0])   # H P- H'
        e = ctx.y - float((H @ theta_pred)[0])  # innovation
        S = hph + ctx.R                          # nominal predictive variance
        ctx.e = e
        ctx.S = S

        w = 1.0 if self.weight_fn is None else float(self.weight_fn.weight(e, S))
        ctx.w = w

        w2 = w * w
        gain = (w2 * P_pred @ H.T) / (w2 * hph + ctx.R)   # (d, 1)
        gain = gain.ravel()

        ctx.theta = theta_pred + gain * e
        eye = np.eye(theta_pred.shape[0])
        ctx.P = (eye - np.outer(gain, H.ravel())) @ P_pred
        ctx.z = e / np.sqrt(S)
        # stash the (weighted) innovation for a lagged adaptive-R estimate next step
        ctx.extra["prev_we"] = w * e
