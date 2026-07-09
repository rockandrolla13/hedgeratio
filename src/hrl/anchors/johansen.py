"""Johansen cointegrating-vector anchor, recomputed on a slow trailing window.

Causal (window strictly before t), normalised so the first (y) element is 1, recomputed on a
monthly-ish cadence, and blended over a short handoff to avoid injecting fake innovations.
Conforms to core.protocols.AnchorProvider.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import AnchorProvider  # noqa: F401  (documents conformance)


def johansen_vector(y: np.ndarray, x: np.ndarray) -> float:
    """Return the (normalised) Johansen hedge ratio beta over a window.

    The cointegrating relation is evec[0]*y + evec[1]*x ~ stationary, so
    y = -(evec[1]/evec[0]) x and beta = -evec[1]/evec[0]. Returns nan on failure so callers
    can fall back to the previous anchor.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 20 or x.size < 20:
        return float("nan")
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        res = coint_johansen(np.column_stack((y, x)), det_order=0, k_ar_diff=1)
        evec = res.evec[:, 0]
        if abs(evec[0]) < 1e-12:
            return float("nan")
        return float(-evec[1] / evec[0])
    except Exception:
        return float("nan")


class JohansenAnchor:
    """Stateful anchor: monthly recompute over a 1-2y window with a linear handoff blend.

    On recompute failure (too few obs, numerical issue) it keeps the previous anchor rather
    than jumping. State is per-path (fresh anchor per grid cell) and picklable.
    """

    def __init__(self, window: int = 504, recompute_every: int = 21,
                 blend_days: int = 5, min_obs: int = 252) -> None:
        self.window = window
        self.recompute_every = recompute_every
        self.blend_days = max(1, blend_days)
        self.min_obs = min_obs
        self._prev: float | None = None
        self._target: float | None = None
        self._t_recompute: int = -(10 ** 9)
        self._t_blend0: int = 0

    def _compute(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        lo = max(0, t - self.window)
        return johansen_vector(y[lo:t], x[lo:t])     # strictly before t (causal)

    def anchor(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        """Return the blended Johansen anchor beta_bar_t (nan until min_obs exist)."""
        if t < self.min_obs:
            return float("nan")

        if self._target is None:
            first = self._compute(t, y, x)
            if not np.isfinite(first):
                return float("nan")
            self._prev = self._target = first
            self._t_recompute = t
            self._t_blend0 = t
        elif t - self._t_recompute >= self.recompute_every:
            fresh = self._compute(t, y, x)
            if np.isfinite(fresh):
                self._prev = self._current(t)
                self._target = fresh
                self._t_recompute = t
                self._t_blend0 = t

        return self._current(t)

    def _current(self, t: int) -> float:
        if self._target is None or self._prev is None:
            return float("nan")
        frac = min(1.0, (t - self._t_blend0) / self.blend_days)
        return float(self._prev + frac * (self._target - self._prev))
