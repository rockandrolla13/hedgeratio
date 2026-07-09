"""Phase 3 experiment: WoLF robust weighting on heavy-tailed / contaminated data (S3).

Monte Carlo over S3 paths comparing vanilla KF vs WoLF-IMQ (and Huber / Student-t references)
on beta RMSE. Reports mean RMSE, paired win-rate vs vanilla, and the median down-weight
applied at the known outlier times. Writes results/phase3_wolf.md.

Run:  python experiments/phase3_wolf.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.pipeline import Pipeline
from hrl.models.linear_ssm import LinearGaussianSSM

N_PATHS = 60
N = 1500
R = 4e-4
CONFIGS = {
    "vanilla": dict(weight_fn="none"),
    "wolf_imq_c3": dict(weight_fn="imq", c=3.0),
    "huber": dict(weight_fn="huber"),
    "student_t": dict(weight_fn="student_t"),
}


def _pipe(**kw) -> Pipeline:
    model = LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5)
    cfg = PipelineConfig(noise_model="fixed", r=R, p0=10.0, **kw)
    return Pipeline.from_config(cfg, model)


def main() -> None:
    rmse = {k: np.empty(N_PATHS) for k in CONFIGS}
    outlier_w = {k: [] for k in CONFIGS}
    for i in range(N_PATHS):
        s = generate("S3", n_steps=N, seed=2000 + i)
        for name, kw in CONFIGS.items():
            res = _pipe(**kw).run(s.y, s.x)
            rmse[name][i] = metrics.beta_rmse(res.beta, s.beta_true)
            if s.outlier_times.size:
                outlier_w[name].append(np.median(res.weights[s.outlier_times]))

    van = rmse["vanilla"]
    rows = []
    for name in CONFIGS:
        r = rmse[name]
        win = float(np.mean(r < van)) if name != "vanilla" else float("nan")
        med_w = float(np.median(outlier_w[name])) if outlier_w[name] else float("nan")
        rows.append(f"| {name} | {r.mean():.4f} | {r.std():.4f} | "
                    f"{'-' if name == 'vanilla' else f'{win:.2f}'} | {med_w:.3f} |")

    report = (
        "# Phase 3 - WoLF robust weighting on S3 (heavy tails + 1% contamination)\n\n"
        f"{N_PATHS} Monte Carlo paths, n={N}, fixed R={R}.\n\n"
        "| config | mean beta RMSE | sd | win-rate vs vanilla | median weight at outliers |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "phase3_wolf.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
