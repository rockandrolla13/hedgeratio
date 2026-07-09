"""Real ETF-pair loaders. Accepts CSVs in data/raw/ -- no hardcoded data vendor.

Canonical replication pair EWA/EWC; also GLD/GDX, an energy pair, and optionally a credit
pair (LQD/IEF or HYG/JNK). Returns aligned log adjusted-close arrays.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def load_pair(y_csv: str | Path, x_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two adjusted-close CSVs, align on date, return (log_y, log_x) float64 arrays."""
    # TODO: read CSVs, inner-join on date, log-transform, raise on schema/alignment failure
    raise NotImplementedError("load_pair")
