"""OracleFilter: a flat, all-inline generalised-Bayes recursion used ONLY by tests.

This is the correctness oracle from the ideate decision: the composed Pipeline must match
this hand-written coupled recursion to ~1e-10 (tests/integration/test_pipeline_vs_oracle.py).
Because everything is in one scope, the coupled R_t / w_t / changepoint dependencies are
trivial to reason about here -- that is the point.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..core.results import FilterResult
from ..models.linear_ssm import LinearGaussianSSM


class OracleFilter:
    """Flat reference implementation of the full composite recursion (WoLF + adaptive-R +
    anchored + changepoint), written inline with no stage abstraction."""

    def __init__(self, cfg: PipelineConfig, model: LinearGaussianSSM) -> None:
        self.cfg = cfg
        self.model = model

    def run(self, y: np.ndarray, x: np.ndarray) -> FilterResult:
        """Run the flat coupled recursion; must equal Pipeline.run to 1e-10."""
        # TODO: inline predict -> R_t -> e,S -> w -> update -> stash -> changepoint
        raise NotImplementedError("OracleFilter.run")
