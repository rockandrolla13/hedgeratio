"""Changepoint stage (step 7) + detectors.

CUSUM/EWMA on signed standardized (robustly-weighted) innovations is the default; a pruned
BOCPD run-length posterior (Adams-MacKay; Fearnhead-Liu) is the research-grade option. On
trigger, inflate P (or reset beta to anchor with wide covariance) and freeze for a refractory
period. NOTE (ideate assumption #2): this stage interacts with the anchor -- test them together.
NOTE (ideate assumption #5): BOCPD MUST prune run-length particles or it is O(t^2).
"""
from __future__ import annotations

from ...core.context import StepContext
from ...core.protocols import ChangepointDetector, StateSpaceModel


class CUSUMDetector:
    """EWMA/CUSUM of signed standardized innovations; threshold calibrated to a target FAR."""

    def __init__(self, threshold: float = 5.0, refractory: int = 20) -> None:
        self.threshold = threshold
        self.refractory = refractory

    def detect(self, ctx: StepContext) -> bool:
        # TODO: accumulate z_t in ctx.extra; trigger on crossing; honour refractory
        raise NotImplementedError("CUSUMDetector.detect")


class BOCPDDetector:
    """Bayesian online changepoint detection with hazard h; pruned run-length posterior."""

    def __init__(self, hazard: float = 1.0 / 500.0, max_run: int = 500) -> None:
        self.hazard = hazard
        self.max_run = max_run   # prune cap -> keeps per-step cost O(max_run), not O(t)

    def detect(self, ctx: StepContext) -> bool:
        # TODO: update run-length posterior (Fearnhead-Liu pruning); trigger on collapse
        raise NotImplementedError("BOCPDDetector.detect")


class ChangepointStage:
    """Step 7: run the detector; on trigger set ctx.reset and inflate ctx.P."""
    name = "changepoint"

    def __init__(self, detector: ChangepointDetector, p_reset_scale: float = 100.0) -> None:
        self.detector = detector
        self.p_reset_scale = p_reset_scale

    def apply(self, ctx: StepContext, model: StateSpaceModel) -> None:
        # TODO: if self.detector.detect(ctx): ctx.reset = True; ctx.P *= p_reset_scale
        raise NotImplementedError("ChangepointStage")
