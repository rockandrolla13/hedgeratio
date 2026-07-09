"""Simple z-score band pairs backtest with costs (secondary metric -- do not overfit).

z = spread / sqrt(S_t); enter at |z| > enter_z, exit at |z| < exit_z; cost_bps per leg,
beta-rehedge costs included. Reports Sharpe, max drawdown, turnover. Real pairs only, as a
sanity layer (the EwanKW cautionary reference: primary claims are estimation/stability).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..config import BacktestConfig


@dataclass
class BacktestResult:
    """Headline trading statistics."""
    sharpe: float
    max_drawdown: float
    turnover: float


class Backtest:
    """z-band strategy over a filtered spread and its innovation-variance path."""

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg

    def run(self, spread: np.ndarray, S: np.ndarray, beta: np.ndarray) -> BacktestResult:
        """Run the band strategy with costs and return trading statistics."""
        # TODO: z = spread/sqrt(S); band entries/exits; PnL net of cost_bps + rehedge costs
        raise NotImplementedError("Backtest.run")
