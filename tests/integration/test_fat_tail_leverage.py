"""Mechanism 5 leverage-point tests (spec T20) plus leverage-statistic diagnostics.

S7 puts fat tails in the REGRESSOR (leverage points); S7b co-times an extreme regressor jump with
a measurement outlier. T20 is a conditional gate: cap the gain in production ONLY if the uncapped
WoLF composite's S7 RMSE exceeds its clean-S1 RMSE by >25% AND the cap recovers >= half the gap
without hurting S1.

Empirically the WoLF composite already survives pure S7 (ratio ~ 0.96 < 1.25), so the gate does NOT
fire and the cap stays off by default -- exactly the documented decision. On S7b (leverage + noise)
WoLF is essential: the vanilla filter's RMSE explodes while the composite stays small.
"""
import numpy as np

from hrl.data.synthetic import generate
from hrl.eval import metrics, stats
from hrl.filters.leverage import (
    H_eff,
    LeverageCappedKalmanRW,
    hph_from_result,
    leverage_statistic,
)

_R = 4e-4
_N = 1500


def _composite(cap):
    return LeverageCappedKalmanRW.wolf(c=3.0, cap=cap, r=_R, p0=10.0)


def _vanilla():
    return LeverageCappedKalmanRW(weight_fn=None, cap=False, r=_R, p0=10.0)


def test_H_eff_caps_high_leverage_only():
    """H_eff leaves ordinary rows untouched and shrinks only above ell_max."""
    H = np.array([1.0, 100.0])
    assert np.allclose(H_eff(H, ell=2.0, ell_max=9.0), H)          # below cap: unchanged
    capped = H_eff(H, ell=36.0, ell_max=9.0)                       # sqrt(9/36) = 0.5
    assert np.allclose(capped, 0.5 * H)
    assert np.allclose(H_eff(H, ell=np.nan), H)                    # undefined leverage: unchanged


def test_leverage_statistic_detects_relative_elevation():
    """The leverage statistic is finite after warm-up, positive, and detects relative elevation.

    For a random-walk regressor the level moves slowly, so leverage elevation is modest (typically
    ell ~ 1.2, occasionally > 3) -- consistent with T20's finding that pure S7 leverage does not
    damage the WoLF composite. We assert the diagnostic is well-formed and picks up the elevation.
    """
    s = generate("S7", n_steps=_N, seed=0)
    res = _composite(cap=False).run(s.y, s.x)
    ell = leverage_statistic(hph_from_result(res, _R), window=250)
    finite = ell[np.isfinite(ell)]
    assert finite.size > 0
    assert np.all(finite > 0.0)
    assert np.all(np.isnan(ell[:249]))                            # NaN until the window fills
    assert np.nanmax(ell) > 1.1                                    # relative elevation is detected


def test_t20_gate_s7_and_s7b():
    """T20 gate: composite not worse than vanilla; measure the S7-vs-S1 damage ratio; S7b needs WoLF."""
    n_paths = 80
    comp7 = np.empty(n_paths)
    cap7 = np.empty(n_paths)
    van7 = np.empty(n_paths)
    comp1 = np.empty(n_paths)
    cap1 = np.empty(n_paths)
    comp7b = np.empty(n_paths)
    van7b = np.empty(n_paths)
    for i in range(n_paths):
        s7 = generate("S7", n_steps=_N, seed=12000 + i)
        s1 = generate("S1", n_steps=_N, seed=13000 + i)
        s7b = generate("S7b", n_steps=_N, seed=14000 + i)
        comp7[i] = metrics.beta_rmse(_composite(False).run(s7.y, s7.x).beta, s7.beta_true)
        cap7[i] = metrics.beta_rmse(_composite(True).run(s7.y, s7.x).beta, s7.beta_true)
        van7[i] = metrics.beta_rmse(_vanilla().run(s7.y, s7.x).beta, s7.beta_true)
        comp1[i] = metrics.beta_rmse(_composite(False).run(s1.y, s1.x).beta, s1.beta_true)
        cap1[i] = metrics.beta_rmse(_composite(True).run(s1.y, s1.x).beta, s1.beta_true)
        comp7b[i] = metrics.beta_rmse(_composite(False).run(s7b.y, s7b.x).beta, s7b.beta_true)
        van7b[i] = metrics.beta_rmse(_vanilla().run(s7b.y, s7b.x).beta, s7b.beta_true)

    # Composite is no worse than vanilla on pure leverage (paired, significant direction).
    _, p7, hl7 = stats.paired_wilcoxon(comp7, van7)
    assert hl7 <= 0.0

    # Damage gate: does uncapped composite S7 RMSE exceed its S1 RMSE by > 25%?
    damage_ratio = comp7.mean() / comp1.mean()
    gate_fires = damage_ratio > 1.25
    assert not gate_fires                                          # documented: gate does NOT fire

    # Cap must not hurt the clean null (non-inferiority within 5%).
    assert cap1.mean() <= 1.05 * comp1.mean()

    # On S7b (leverage coincident with a measurement outlier) WoLF is essential.
    _, p7b, hl7b = stats.paired_wilcoxon(comp7b, van7b)
    assert hl7b < 0.0 and p7b < 1e-3
    assert comp7b.mean() < 0.5 * van7b.mean()
