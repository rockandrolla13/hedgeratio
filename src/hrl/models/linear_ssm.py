"""Linear-Gaussian state space: state = (alpha, beta), observation y = alpha + beta*x.

Default transition is a random walk (F = I, Q = diag(q_alpha, q_beta)); the AnchoredTransition
stage overrides the beta transition at run time. Conforms to core.protocols.StateSpaceModel.
"""
from __future__ import annotations

import numpy as np

from ..core.context import StepContext


class LinearGaussianSSM:
    """y_t = alpha_t + beta_t x_t + eps_t; state theta = (alpha, beta)."""

    dim: int = 2

    def __init__(self, q_alpha: float = 1e-7, q_beta: float = 1e-5) -> None:
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self._F = np.eye(2)
        self._Q = np.diag([self.q_alpha, self.q_beta])

    def transition(self, ctx: StepContext) -> tuple[np.ndarray, np.ndarray]:
        """Random-walk transition: F = I_2, Q = diag(q_alpha, q_beta)."""
        return self._F, self._Q

    def observation(self, ctx: StepContext) -> np.ndarray:
        """H_t = [1, x_t] as a (1, 2) row."""
        return np.array([[1.0, ctx.x]])
