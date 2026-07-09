"""Synthetic DGPs with known ground truth (design/plan Phase 2).

S1 static | S2 slow drift | S3 heavy tails/contamination | S4 stochastic vol |
S5 structural breaks | S6 partial cointegration. x_t is a scaled random walk (log-price-like).
Each generator is seeded and returns a SyntheticSample.

Phase 1 implements S1-S3 (per plan section 11 step 1). S4-S6 remain stubs.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

# Shared scales so the DGPs are comparable.
_ALPHA = 0.5
_BETA = 1.2
_EPS_SD = 0.02       # Gaussian observation-noise sd
_X_START = 4.6       # log-price-like level (~100)
_X_STEP_SD = 0.02    # random-walk increment sd


@dataclass
class SyntheticSample:
    """One simulated path with its ground truth."""
    y: np.ndarray
    x: np.ndarray
    beta_true: np.ndarray
    alpha_true: np.ndarray
    break_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    outlier_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def _x_series(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate x_t as a scaled random walk at realistic log-price-like levels."""
    return _X_START + np.cumsum(rng.normal(0.0, _X_STEP_SD, n))


def s1_static(n: int, rng: np.random.Generator) -> SyntheticSample:
    """Constant beta, Gaussian noise. Sanity check."""
    x = _x_series(n, rng)
    alpha_true = np.full(n, _ALPHA)
    beta_true = np.full(n, _BETA)
    y = alpha_true + beta_true * x + rng.normal(0.0, _EPS_SD, n)
    return SyntheticSample(y=y, x=x, beta_true=beta_true, alpha_true=alpha_true)


def s2_drift(n: int, rng: np.random.Generator) -> SyntheticSample:
    """Beta follows a slow sinusoid; Gaussian noise. Genuine (non-break) drift."""
    x = _x_series(n, rng)
    t = np.arange(n)
    beta_true = _BETA + 0.3 * np.sin(2.0 * np.pi * t / n)
    alpha_true = np.full(n, _ALPHA)
    y = alpha_true + beta_true * x + rng.normal(0.0, _EPS_SD, n)
    return SyntheticSample(y=y, x=x, beta_true=beta_true, alpha_true=alpha_true)


def s3_heavy(n: int, rng: np.random.Generator, nu: float = 4.0,
             contam: float = 0.01, contam_scale: float = 10.0) -> SyntheticSample:
    """Student-t(nu) observation noise plus ~1% additive contamination (mirrors the WoLF setup)."""
    x = _x_series(n, rng)
    alpha_true = np.full(n, _ALPHA)
    beta_true = np.full(n, _BETA)
    # Student-t scaled to match _EPS_SD in the bulk.
    t_scale = _EPS_SD / np.sqrt(nu / (nu - 2.0))
    eps = t_scale * rng.standard_t(nu, n)
    outlier_mask = rng.random(n) < contam
    eps[outlier_mask] += rng.normal(0.0, contam_scale * _EPS_SD, outlier_mask.sum())
    y = alpha_true + beta_true * x + eps
    return SyntheticSample(y=y, x=x, beta_true=beta_true, alpha_true=alpha_true,
                           outlier_times=np.flatnonzero(outlier_mask))


def s4_stochvol(n: int, rng: np.random.Generator) -> SyntheticSample:
    """GARCH(1,1) or two-state (calm/stress x5) volatility. (Phase 4.)"""
    raise NotImplementedError("s4_stochvol")


def s5_breaks(n: int, rng: np.random.Generator) -> SyntheticSample:
    """Piecewise-constant beta with jumps at known times. (Phase 6.)"""
    raise NotImplementedError("s5_breaks")


def s6_pci(n: int, rng: np.random.Generator) -> SyntheticSample:
    """Spread = random walk + AR(1)(rho ~ 0.9) (Clegg-Krauss): the PCI temptation. (Phase 7.)"""
    raise NotImplementedError("s6_pci")


_REGISTRY = {
    "S1": s1_static, "S2": s2_drift, "S3": s3_heavy,
    "S4": s4_stochvol, "S5": s5_breaks, "S6": s6_pci,
}


def generate(name: str, n_steps: int, seed: int) -> SyntheticSample:
    """Dispatch to a named DGP with a deterministic seed."""
    rng = np.random.default_rng(seed)
    return _REGISTRY[name](n_steps, rng)
