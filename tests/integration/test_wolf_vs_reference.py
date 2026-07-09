"""Integration: our scalar WoLF-IMQ update vs the reference implementation.

Cross-check against github.com/gerdm/weighted-likelihood-filter on one linear example.
Match to ~1e-6 in float64 (document any IMQ normalization differences).
"""
import pytest

from hrl.filters.stages.wolf import IMQWeight


def test_imq_importable():
    """Smoke: the IMQ weight function is importable."""
    assert IMQWeight(c=2.0).c == 2.0


@pytest.mark.skip(reason="TODO: pin reference vectors from gerdm repo; assert allclose atol=1e-6")
def test_wolf_matches_reference():
    """Our weighted update reproduces the reference repo on identical inputs."""
    # TODO: load a saved reference (e, S, w) triple; compare weights and posterior update
    ...
