# hedge_ratio_lab (`hrl`)

Robust, outlier-tolerant, time-varying **hedge-ratio estimation** for mid-frequency pairs
trading, built as a **config-composed generalised-Bayes Kalman pipeline** plus a parallel
ablation harness that *attributes* each robustness property to a specific component.

## Core idea

The four robustness components are **not** orthogonal switches — they are a fixed-order,
coupled recursion over a shared `StepContext`, driven by `filters.pipeline.Pipeline`:

```
predict -> R_t (adaptive) -> innovation e,S -> weight w -> weighted update -> standardize/stash -> changepoint reset
```

The **ablation grid == which stages are present**, which is why the architecture and the
experiment are the same object.

## Layout

```
src/hrl/
  core/        StepContext, protocols, FilterResult, parallel_map
  models/      LinearGaussianSSM, PartialCointegrationSSM
  filters/     Pipeline, baselines (rolling OLS/TLS, RW-KF), oracle (test-only), stages/
  anchors/     Johansen, TLS slow anchors
  data/        synthetic DGPs (S1..S6), real ETF loaders
  eval/        metrics, backtest, reports
  experiments/ AblationRunner, typer CLI
```

## Quickstart

```bash
pip install -e ".[dev]"
hrl run --config config/default.yaml
pytest
```

Everything is online/causal and deterministic (seeded). Hyperparameters are fit on the first
40% of data; a single global set is used across pairs. See `core/context.py` for the full
mathematical specification.
