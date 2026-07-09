"""Smoke + invariant tests for the changepoint stage and detectors."""
from hrl.filters.stages.changepoint import BOCPDDetector, ChangepointStage, CUSUMDetector


def test_import_and_instantiate():
    """Smoke: detectors and stage construct."""
    assert CUSUMDetector(threshold=5.0).threshold == 5.0
    assert BOCPDDetector().max_run == 500      # prune cap present (ideate #5)
    assert ChangepointStage(CUSUMDetector()).name == "changepoint"


# TODO: on S5, measure detection delay at fixed FAR and post-break reconvergence time vs
#   vanilla KF and inflated-Q ("desk fix").
# TODO: required result -- changepoint reset reconverges faster than inflated-Q at equal
#   steady-state beta noise (decoupling).
# TODO: BOCPD per-step cost stays O(max_run), not O(t) (pruning works).
