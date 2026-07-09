"""Johansen cointegrating-vector anchor, recomputed on a slow trailing window.

Causal (window strictly before t), normalised so the first element is 1, recomputed monthly,
and blended over ~5 days on handoff to avoid injecting fake innovations. Conforms to
core.protocols.AnchorProvider.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import AnchorProvider  # noqa: F401  (documents conformance)


def johansen_vector(y: np.ndarray, x: np.ndarray) -> float:
    """Return the (normalised) Johansen hedge ratio over a window. Falls back on failure."""
    # TODO: statsmodels.tsa.vector_ar.vecm.coint_johansen; normalise first element to 1
    raise NotImplementedError("johansen_vector")


class JohansenAnchor:
    """Stateful anchor: monthly recompute over a 1-2y window with 5-day handoff blend."""

    def __init__(self, window: int = 504, recompute_every: int = 21,
                 blend_days: int = 5, min_obs: int = 252) -> None:
        self.window = window
        self.recompute_every = recompute_every
        self.blend_days = blend_days
        self.min_obs = min_obs

    def anchor(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        """beta_bar_t from the last recompute, linearly blended toward a fresh estimate."""
        # TODO: recompute on cadence over y[:t], x[:t]; blend; fall back if < min_obs
        raise NotImplementedError("JohansenAnchor.anchor")
