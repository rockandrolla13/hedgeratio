"""Smoke + invariant tests for WoLF weighting."""
from hrl.filters.stages.wolf import HuberWeight, IMQWeight, StudentTWeight, WolfReweightStage


def test_import_and_instantiate():
    """Smoke: weight functions and stage construct."""
    assert IMQWeight(c=3.0).c == 3.0
    assert HuberWeight() is not None
    assert StudentTWeight() is not None
    assert WolfReweightStage(IMQWeight()).name == "wolf"


# TODO: c -> inf recovers vanilla KF (w == 1.0) -- assert numerically.
# TODO (hypothesis): bounded influence -- |theta - theta_pred| saturates as |e| -> inf
#   under WoLF, but grows linearly under vanilla KF.
# TODO: on S3, WoLF beta RMSE < vanilla KF beta RMSE at >=95% MC confidence.
