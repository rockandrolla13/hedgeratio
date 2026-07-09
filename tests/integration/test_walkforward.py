"""Integration: walk-forward protocol on real pairs.

Fit hyperparameters (Q, R or lam, c, phi, thresholds) on the first 40% only; everything after
is untouched evaluation. One global hyperparameter set across pairs (no per-pair tuning).
"""
import pytest

from hrl.experiments.runner import AblationRunner


def test_runner_importable():
    """Smoke: the ablation runner is importable."""
    assert AblationRunner is not None


@pytest.mark.skip(reason="TODO: needs data/raw/ fixtures and filled-in stages")
def test_walkforward_no_lookahead():
    """No hyperparameter fit uses data beyond the first 40% of any pair."""
    # TODO: assert the fit window never reads indices >= 0.4 * T
    ...
