"""Total least squares (errors-in-variables) anchor.

TLS avoids the OLS attenuation bias under microstructure noise: OLS(y|x) and OLS(x|y) give
inconsistent hedge ratios, TLS does not. Conforms to core.protocols.AnchorProvider.

The provider is causal (uses only observations strictly before t), recomputes on a slow
cadence over a trailing window, and linearly blends from the old to the new estimate over a
short handoff so anchor steps do not enter the filter as fake innovations.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import AnchorProvider  # noqa: F401  (documents conformance)


def tls_fit(y: np.ndarray, x: np.ndarray) -> float:
    """Orthogonal-regression hedge ratio via SVD of the centered [x, y] matrix.

    Returns the slope of the least-variance direction, i.e. the total-least-squares estimate
    of beta in y ~ beta x. Returns nan if the window is degenerate.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return float("nan")
    m = np.column_stack((x - x.mean(), y - y.mean()))
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    vx, vy = vt[-1]                 # direction of least variance
    if abs(vy) < 1e-12:
        return float("nan")
    return float(-vx / vy)


class TlsAnchor:
    """Stateful TLS anchor: slow trailing-window recompute with a linear handoff blend.

    State lives on the instance and is per-path (the runner builds a fresh anchor per grid
    cell), so it stays picklable and free of cross-path leakage.
    """

    def __init__(self, window: int = 504, recompute_every: int = 21,
                 blend_days: int = 5, min_obs: int = 252) -> None:
        self.window = window
        self.recompute_every = recompute_every
        self.blend_days = max(1, blend_days)
        self.min_obs = min_obs
        self._prev: float | None = None     # value we are blending from
        self._target: float | None = None   # freshly computed value we blend toward
        self._t_recompute: int = -(10 ** 9)
        self._t_blend0: int = 0

    def _compute(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        lo = max(0, t - self.window)
        return tls_fit(y[lo:t], x[lo:t])     # strictly before t (causal)

    def anchor(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        """Return the blended TLS anchor beta_bar_t (nan until min_obs observations exist)."""
        if t < self.min_obs:
            return float("nan")

        if self._target is None:
            first = self._compute(t, y, x)
            self._prev = self._target = first
            self._t_recompute = t
            self._t_blend0 = t
        elif t - self._t_recompute >= self.recompute_every:
            fresh = self._compute(t, y, x)
            if np.isfinite(fresh):
                self._prev = self._current(t)   # blend starts from where we are now
                self._target = fresh
                self._t_recompute = t
                self._t_blend0 = t

        return self._current(t)

    def _current(self, t: int) -> float:
        if self._target is None or self._prev is None:
            return float("nan")
        frac = min(1.0, (t - self._t_blend0) / self.blend_days)
        return float(self._prev + frac * (self._target - self._prev))
