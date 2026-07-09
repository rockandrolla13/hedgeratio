"""Partial-cointegration state space (Clegg-Krauss 2018), phase 7.

Joint state (alpha, beta, m, r) where the spread decomposes into a mean-reverting
component m_t = rho m_{t-1} + xi_t plus a random walk r_t = r_{t-1} + eta_t. Purpose: give
permanent spread shocks somewhere to go other than beta. (rho, var_xi, var_eta) fit by MLE
on the training split. Conforms to core.protocols.StateSpaceModel.
"""
from __future__ import annotations

import numpy as np

from ..core.context import StepContext


class PartialCointegrationSSM:
    """State theta = (alpha, beta, m, r); spread = m (mean-reverting) + r (random walk)."""

    dim: int = 4

    def __init__(self, rho: float = 0.9, var_xi: float = 1e-4,
                 var_eta: float = 1e-5, q_beta: float = 1e-5) -> None:
        self.rho = rho
        self.var_xi = var_xi
        self.var_eta = var_eta
        self.q_beta = q_beta

    def transition(self, ctx: StepContext) -> tuple[np.ndarray, np.ndarray]:
        """F blends RW(alpha, beta, r) with AR(1) on m; Q from (q_beta, var_xi, var_eta)."""
        # TODO: build 4x4 F with rho on the m row; Q diag with var_xi, var_eta
        raise NotImplementedError("PCI transition")

    def observation(self, ctx: StepContext) -> np.ndarray:
        """H_t = [1, x_t, 1, 1]: y = alpha + beta*x + m + r."""
        # TODO: return np.array([[1.0, ctx.x, 1.0, 1.0]])
        raise NotImplementedError("PCI observation row")
