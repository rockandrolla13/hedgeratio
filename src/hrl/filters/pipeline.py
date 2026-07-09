"""Pipeline: orders and drives StepStages over one sample; enforces legal compositions.

Canonical stage order (the data-flow contract that keeps the coupled components correct):
    1 predict            (plain or anchored: theta-, P- from F, Q)
    2 adaptive R_t       (plug-in EWMA is lagged => no within-step circularity)
    3-5 update           (innovation e, S -> robust weight w -> weighted gain)  [GaussianUpdateStage]
    6 (stash happens inside the update stage)
    7 changepoint detect -> P reset

Composition rules enforced at construction:
  * exactly one transition/predict stage (plain XOR anchored);
  * VB-AKF noise model and WoLF weighting are mutually exclusive within a step.

Phase 1 implements the vanilla path (predict + update, fixed R, no weight/anchor/changepoint).
Later phases extend `from_config`; unsupported toggles raise NotImplementedError rather than
being silently ignored.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..core.context import StepContext
from ..core.protocols import StateSpaceModel, StepStage
from ..core.results import FilterResult
from .stages.gaussian import GaussianUpdateStage, PredictStage
from .stages.wolf import make_weight_fn

_TRANSITION_STAGES = {"predict"}


class Pipeline:
    """Holds an ordered list of stages and runs the recursion over a full sample."""

    def __init__(self, stages: list[StepStage], model: StateSpaceModel,
                 r0: float = 1e-3, p0: float = 1.0) -> None:
        self.stages = stages
        self.model = model
        self.r0 = float(r0)
        self.p0 = float(p0)
        self._validate()

    def _validate(self) -> None:
        """Reject illegal compositions (see module docstring)."""
        transitions = [s for s in self.stages if getattr(s, "name", None) in _TRANSITION_STAGES]
        if len(transitions) != 1:
            raise ValueError(
                f"pipeline needs exactly one transition stage, found {len(transitions)}"
            )
        if self.stages[0] is not transitions[0]:
            raise ValueError("the transition stage must run first")

    @classmethod
    def from_config(cls, cfg: PipelineConfig, model: StateSpaceModel) -> "Pipeline":
        """Build a Pipeline (stages in canonical order) from a PipelineConfig."""
        if cfg.anchor != "none":
            raise NotImplementedError("anchored transition arrives in Phase 5")
        if cfg.noise_model != "fixed":
            raise NotImplementedError("adaptive R arrives in Phase 4")
        if cfg.detector != "none":
            raise NotImplementedError("changepoint layer arrives in Phase 6")

        weight_fn = make_weight_fn(cfg.weight_fn, cfg)
        stages: list[StepStage] = [PredictStage(), GaussianUpdateStage(weight_fn=weight_fn)]
        return cls(stages, model, r0=cfg.r, p0=cfg.p0)

    def run(self, y, x) -> FilterResult:
        """Filter the (y, x) arrays and return the per-step FilterResult paths."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if y.shape != x.shape:
            raise ValueError(f"y and x must align; got {y.shape} vs {x.shape}")
        n = y.shape[0]
        d = self.model.dim

        ctx = StepContext(
            t=0, y=float(y[0]), x=float(x[0]),
            theta=np.zeros(d), P=np.eye(d) * self.p0, R=self.r0,
        )

        beta = np.empty(n)
        alpha = np.empty(n)
        spread = np.empty(n)   # innovation e_t (predictive residual; pairs with S_t)
        S = np.empty(n)
        weights = np.empty(n)
        resets = np.zeros(n, dtype=bool)

        for t in range(n):
            ctx.t = t
            ctx.y = float(y[t])
            ctx.x = float(x[t])
            ctx.R = self.r0
            ctx.w = 1.0
            ctx.reset = False

            for stage in self.stages:
                stage.apply(ctx, self.model)

            alpha[t] = ctx.theta[0]
            beta[t] = ctx.theta[1]
            spread[t] = ctx.e
            S[t] = ctx.S
            weights[t] = ctx.w
            resets[t] = ctx.reset

        return FilterResult(beta=beta, alpha=alpha, spread=spread,
                            S=S, weights=weights, resets=resets)
