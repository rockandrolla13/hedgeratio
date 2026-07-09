"""StepContext and the mathematical specification of the estimator.

Model (design Phase 1.5):
    Observation : y_t = alpha_t + beta_t x_t + eps_t,  eps_t heavy-tailed, scale R_t,  H_t = [1, x_t]
    State       : alpha_t = alpha_{t-1} + w_alpha
                  beta_t  = beta_bar_t + phi (beta_{t-1} - beta_bar_t) + w_beta,  phi in [0.98, 0.999]
    GB update   : w_t = (1 + e_t^2 / (c^2 S_t))^(-1/2)              (WoLF-IMQ)
                  K_t = w_t^2 P- H' / (w_t^2 H P- H' + R_t)
    Guards      : R_t >= R_floor; Johansen needs >= min_obs else fall back to prior anchor; phi < 1.

Canonical stage order (owned by filters.pipeline.Pipeline):
    predict -> R_t -> innovation e,S -> weight w -> weighted update -> standardize/stash -> changepoint
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepContext:
    """Mutable carrier for one filter step.

    Fields are consistent only AFTER the full canonical stage order has run. `extra`
    holds per-path policy scratch (vol state, run-length posterior, anchor cadence)
    and must remain picklable for parallel execution.
    """
    t: int
    y: float
    x: float
    theta: np.ndarray            # state mean, shape (d,) -> (alpha, beta[, m, r])
    P: np.ndarray                # state covariance, shape (d, d)
    R: float                     # measurement-noise variance R_t (>= R_floor)
    e: float = 0.0               # innovation  y - H @ theta_pred
    S: float = 1.0               # innovation variance  H P- H' + R
    w: float = 1.0               # robust weight in (0, 1]; 1.0 == vanilla Gaussian
    z: float = 0.0               # standardized innovation  e / sqrt(S)
    reset: bool = False          # changepoint declared this step
    extra: dict = field(default_factory=dict)
