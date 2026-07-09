"""Smoke + invariant tests for the adaptive measurement-noise stage."""
from hrl.filters.stages.adaptive_r import AdaptiveRStage, EWMANoise, GARCHNoise, VBAKFNoise


def test_import_and_instantiate():
    """Smoke: noise models and stage construct."""
    assert EWMANoise(lam=0.94).lam == 0.94
    assert GARCHNoise() is not None
    assert VBAKFNoise().n_iter == 3
    assert AdaptiveRStage(EWMANoise()).name == "adaptive_r"


# TODO: on S4, filtered-beta variance in stress regimes falls vs fixed-R KF, without
#   degrading calm-regime tracking (counter-cyclical gain).
# TODO: VB-AKF fixed point converges within n_iter on synthetic data.
