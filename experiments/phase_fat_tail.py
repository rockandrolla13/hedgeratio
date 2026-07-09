"""Fat-tail robustness layer experiment: assemble the implementable mechanisms into the report.

Runs the parts of the spec that do NOT depend on Phases 4/6 (Mechanisms 1, 4, 5) over a small
Monte Carlo and writes results/FAT_TAIL_REPORT.md via hrl.eval.fat_tail_report. Mechanisms 2 and 3
render as PENDING placeholders until adaptive-R (Phase 4) and changepoint (Phase 6) land.

Run:  PYTHONPATH=src python experiments/phase_fat_tail.py
"""
from __future__ import annotations

import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import metrics, stats
from hrl.eval.conformal_bands import aci_bands, gaussian_bands
from hrl.eval.fat_tail_report import write_fat_tail_report
from hrl.filters.leverage import LeverageCappedKalmanRW
from hrl.filters.pipeline import Pipeline
from hrl.filters.student_t import StudentTKalmanRW
from hrl.models.linear_ssm import LinearGaussianSSM

R = 4e-4
N = 1500
N_PATHS = 60


def _pipe(**kw) -> Pipeline:
    cfg = PipelineConfig(noise_model="fixed", r=R, p0=10.0, **kw)
    return Pipeline.from_config(cfg, LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5))


def mech1() -> dict:
    """T3 (WoLF vs vanilla on S3) and T4 (efficiency on S1)."""
    van3 = np.empty(N_PATHS)
    wolf3 = np.empty(N_PATHS)
    van1 = np.empty(N_PATHS)
    wolf1 = np.empty(N_PATHS)
    t1 = np.empty(N_PATHS)
    for i in range(N_PATHS):
        s3 = generate("S3", N, 9000 + i)
        van3[i] = metrics.beta_rmse(_pipe(weight_fn="none").run(s3.y, s3.x).beta, s3.beta_true)
        wolf3[i] = metrics.beta_rmse(_pipe(weight_fn="imq", c=3.0).run(s3.y, s3.x).beta, s3.beta_true)
        s1 = generate("S1", N, 7000 + i)
        van1[i] = metrics.beta_rmse(_pipe(weight_fn="none").run(s1.y, s1.x).beta, s1.beta_true)
        wolf1[i] = metrics.beta_rmse(_pipe(weight_fn="imq", c=3.0).run(s1.y, s1.x).beta, s1.beta_true)
        t1[i] = metrics.beta_rmse(StudentTKalmanRW(nu=8.0, r=R, p0=10.0).run(s1.y, s1.x).beta, s1.beta_true)
    _, p, hl = stats.paired_wilcoxon(wolf3, van3)
    return {
        "T3 WoLF vs vanilla (S3)": {"vanilla RMSE": float(van3.mean()),
                                    "WoLF RMSE": float(wolf3.mean()),
                                    "wilcoxon p": float(p), "H-L shift": float(hl),
                                    "win rate": float(np.mean(wolf3 < van3))},
        "T4 efficiency (S1)": {"RE WoLF": stats.relative_efficiency(wolf1, van1),
                               "RE t(8)": stats.relative_efficiency(t1, van1)},
    }


def mech4() -> dict:
    """T15 unconditional coverage: ACI vs Gaussian-S Kupiec fail-to-reject rates."""
    alpha = 0.10
    aci_ftr = gauss_ftr = 0
    aci_cov = np.empty(N_PATHS)
    gauss_cov = np.empty(N_PATHS)
    for i in range(N_PATHS):
        s = generate("S3", 2000, 11000 + i)
        r = _pipe(weight_fn="none").run(s.y, s.x)
        a = aci_bands(r.spread, r.S, alpha=alpha, gamma=0.01)
        g = gaussian_bands(r.spread, r.S, alpha=alpha)
        aci_cov[i], gauss_cov[i] = a["coverage"], g["coverage"]
        aci_ftr += stats.kupiec_pof(int(a["err"].sum()), a["n_eval"], alpha)[1] > 0.05
        gauss_ftr += stats.kupiec_pof(int(g["err"].sum()), g["n_eval"], alpha)[1] > 0.05
    return {
        "T15 ACI (conformal)": {"mean coverage": float(aci_cov.mean()),
                                "Kupiec fail-to-reject frac": aci_ftr / N_PATHS,
                                "nominal": 1 - alpha},
        "T15 Gaussian-S band": {"mean coverage": float(gauss_cov.mean()),
                                "Kupiec fail-to-reject frac": gauss_ftr / N_PATHS,
                                "nominal": 1 - alpha},
    }


def mech5() -> dict:
    """T20 leverage-damage gate on S7 / S7b."""
    comp7 = np.empty(N_PATHS)
    comp1 = np.empty(N_PATHS)
    comp7b = np.empty(N_PATHS)
    van7b = np.empty(N_PATHS)
    for i in range(N_PATHS):
        s7 = generate("S7", N, 12000 + i)
        s1 = generate("S1", N, 13000 + i)
        s7b = generate("S7b", N, 14000 + i)
        comp7[i] = metrics.beta_rmse(LeverageCappedKalmanRW.wolf(c=3.0, cap=False, r=R, p0=10.0).run(s7.y, s7.x).beta, s7.beta_true)
        comp1[i] = metrics.beta_rmse(LeverageCappedKalmanRW.wolf(c=3.0, cap=False, r=R, p0=10.0).run(s1.y, s1.x).beta, s1.beta_true)
        comp7b[i] = metrics.beta_rmse(LeverageCappedKalmanRW.wolf(c=3.0, cap=False, r=R, p0=10.0).run(s7b.y, s7b.x).beta, s7b.beta_true)
        van7b[i] = metrics.beta_rmse(LeverageCappedKalmanRW(cap=False, r=R, p0=10.0).run(s7b.y, s7b.x).beta, s7b.beta_true)
    ratio = float(comp7.mean() / comp1.mean())
    return {
        "T20 gate (S7 vs S1)": {"composite S7 RMSE": float(comp7.mean()),
                                "composite S1 RMSE": float(comp1.mean()),
                                "damage ratio": ratio,
                                "gate fires (>1.25)": ratio > 1.25},
        "T20 S7b (leverage+noise)": {"composite RMSE": float(comp7b.mean()),
                                     "vanilla RMSE": float(van7b.mean())},
    }


def main() -> None:
    results = {"mech1": mech1(), "mech4": mech4(), "mech5": mech5()}
    path = write_fat_tail_report(results, "results")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
