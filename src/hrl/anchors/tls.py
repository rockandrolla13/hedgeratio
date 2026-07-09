"""Total least squares (errors-in-variables) anchor.

TLS avoids the OLS attenuation bias under microstructure noise: OLS(y|x) and OLS(x|y) give
inconsistent hedge ratios, TLS does not. Conforms to core.protocols.AnchorProvider.
"""
from __future__ import annotations

import numpy as np

from ..core.protocols import AnchorProvider  # noqa: F401  (documents conformance)


def tls_fit(y: np.ndarray, x: np.ndarray) -> float:
    """Orthogonal-regression hedge ratio via SVD of the centered [x, y] matrix."""
    # TODO: total least squares slope from the smallest right singular vector
    raise NotImplementedError("tls_fit")


class TlsAnchor:
    """Stateful TLS anchor with the same cadence + handoff-blend contract as JohansenAnchor."""

    def __init__(self, window: int = 504, recompute_every: int = 21,
                 blend_days: int = 5, min_obs: int = 252) -> None:
        self.window = window
        self.recompute_every = recompute_every
        self.blend_days = blend_days
        self.min_obs = min_obs

    def anchor(self, t: int, y: np.ndarray, x: np.ndarray) -> float:
        """beta_bar_t from a slow trailing-window TLS fit, blended on handoff."""
        # TODO: recompute on cadence over y[:t], x[:t]; blend; fall back if < min_obs
        raise NotImplementedError("TlsAnchor.anchor")
