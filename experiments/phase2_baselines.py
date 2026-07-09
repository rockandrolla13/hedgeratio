"""Phase 2 experiment: baselines + EM/MLE fit, validated on S1/S2.

Fits the vanilla KF on the first 40% of each series (walk-forward discipline), reports the
fitted (q, r), the implied EWRLS half-life, and out-of-sample estimation metrics vs rolling
OLS/TLS. Writes a markdown table to results/phase2_baselines.md.

Run:  python experiments/phase2_baselines.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.baselines import VanillaKalmanRW, ewma_half_life, rolling_ols, rolling_tls

N = 2500
WINDOW = 120
SEED = 0
DGPS = ["S1", "S2"]


def _oos_rmse(beta: np.ndarray, truth: np.ndarray, cut: int) -> float:
    b = beta[cut:]
    t = truth[cut:]
    mask = ~np.isnan(b)
    return metrics.beta_rmse(b[mask], t[mask])


def main() -> None:
    rows: list[str] = []
    for name in DGPS:
        s = generate(name, n_steps=N, seed=SEED)
        cut = int(0.4 * N)

        kf = VanillaKalmanRW(q_alpha=1e-6, q_beta=1e-4, r=1e-2).fit(s.y[:cut], s.x[:cut])
        res = kf.run(s.y, s.x)
        hl = ewma_half_life(kf.q_beta, kf.r)

        ols = rolling_ols(s.y, s.x, WINDOW)
        tls = rolling_tls(s.y, s.x, WINDOW)

        rows.append(
            f"| {name} | {kf.q_beta:.2e} | {kf.r:.2e} | {hl:.0f} | "
            f"{_oos_rmse(res.beta, s.beta_true, cut):.4f} | "
            f"{_oos_rmse(ols, s.beta_true, cut):.4f} | "
            f"{_oos_rmse(tls, s.beta_true, cut):.4f} | "
            f"{metrics.delta_beta_var(res.beta[cut:]):.2e} | "
            f"{metrics.delta_beta_var(ols[~np.isnan(ols)]):.2e} |"
        )

    header = (
        "# Phase 2 - Baselines validation (S1/S2)\n\n"
        "Fit on first 40%; metrics are out-of-sample (last 60%).\n\n"
        "| DGP | fit q_beta | fit r | EWMA half-life | KF RMSE | OLS RMSE | TLS RMSE "
        "| KF churn | OLS churn |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    report = header + "\n".join(rows) + "\n"
    (out_dir / "phase2_baselines.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
