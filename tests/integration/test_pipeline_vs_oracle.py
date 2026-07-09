"""Integration: the composed Pipeline must equal the flat OracleFilter to ~1e-10.

This is the correctness gate from the ideate decision -- it defends against the "orthogonal
switches" abstraction silently drifting from the true coupled recursion.
"""
import pytest

from hrl.filters.oracle import OracleFilter
from hrl.filters.pipeline import Pipeline


def test_symbols_importable():
    """Smoke: both the pipeline and the oracle are importable."""
    assert Pipeline is not None
    assert OracleFilter is not None


@pytest.mark.skip(reason="TODO: implement once stages + oracle are filled in")
def test_pipeline_matches_oracle():
    """Full composite (WoLF + adaptive-R + anchored + changepoint) equals oracle to 1e-10."""
    # TODO: build both from the same PipelineConfig; run on one S5+S3 path; allclose atol=1e-10
    ...
