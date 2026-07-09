"""Load-bearing contracts for the hedge-ratio filtering pipeline.

Concrete implementations live in models/, filters/stages/, and anchors/. Every Protocol
here has >=2 concrete variants (see the ablation reference methods), which is why they are
protocols and not concrete classes.
"""
from __future__ import annotations
from typing import Protocol

import numpy as np

from .context import StepContext


class StateSpaceModel(Protocol):
    """Supplies the linear-Gaussian pieces for one step; swap linear <-> PCI freely."""
    dim: int
    def transition(self, ctx: StepContext) -> tuple[np.ndarray, np.ndarray]:
        """Return (F, Q): transition matrix and process-noise covariance at t."""
        ...
    def observation(self, ctx: StepContext) -> np.ndarray:
        """Return H_t, the (1, dim) observation row for the current x_t."""
        ...


class StepStage(Protocol):
    """One in-place transform on the context. Stages run in the Pipeline's fixed order,
    hold no cross-path mutable state, and must be picklable."""
    name: str
    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None: ...


class WeightFunction(Protocol):
    """Bounded-influence observation weight; IMQ (primary), Huber, Student-t."""
    def weight(self, e: float, S: float) -> float: ...   # returns w in (0, 1]


class MeasurementNoiseModel(Protocol):
    """R_t policy: EWMA plug-in (primary), GARCH(1,1), VB-AKF (inverse-gamma)."""
    def update(self, ctx: StepContext) -> float: ...      # returns R_t


class AnchorProvider(Protocol):
    """Slow, causal hedge-ratio anchor beta_bar_t (Johansen / TLS) with handoff blend."""
    def anchor(self, t: int, y: np.ndarray, x: np.ndarray) -> float: ...


class ChangepointDetector(Protocol):
    """CUSUM/EWMA (default) or pruned BOCPD run-length; True => trigger P-reset."""
    def detect(self, ctx: StepContext) -> bool: ...
