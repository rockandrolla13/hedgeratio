"""FilterResult: per-run output arrays consumed by eval/."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class FilterResult:
    """Float64 paths produced by one Pipeline run over one sample."""
    beta: np.ndarray        # beta_hat_t
    alpha: np.ndarray       # alpha_hat_t
    spread: np.ndarray      # y_t - H_t theta_t (filtered residual)
    S: np.ndarray           # innovation variance path (for z-scores / PIT)
    weights: np.ndarray     # robust weights w_t
    resets: np.ndarray      # bool changepoint flags
