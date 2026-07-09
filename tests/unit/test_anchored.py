"""Smoke + invariant tests for the AR(1)-anchored transition stage and anchors."""
from hrl.anchors.johansen import JohansenAnchor
from hrl.anchors.tls import TlsAnchor
from hrl.filters.stages.anchored import AnchoredTransitionStage


def test_import_and_instantiate():
    """Smoke: anchors and anchored stage construct; stage occupies the transition slot."""
    assert AnchoredTransitionStage(JohansenAnchor(), phi=0.99).name == "predict"
    assert TlsAnchor() is not None


# TODO: on S2 (genuine drift) anchored filter does not lag catastrophically (RMSE within
#   tolerance of RW-KF).
# TODO: on S6 (PCI temptation) anchored beta variance materially below RW-KF while filtered
#   spread stationarity (ADF) improves or holds.
# TODO (ideate #2): anchor vs changepoint interaction -- test them together, not just WoLF/BOCPD.
