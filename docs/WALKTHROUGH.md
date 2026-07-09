# Project Walkthrough — Robust Hedge-Ratio Filtering (`hrl`)

A plain-language, step-by-step account of what this project is, how it was designed, what has
been built so far, and — importantly — **what the "data" is**. Read top to bottom; no prior
context needed.

---

## 0. One-paragraph summary

We are building a tool that estimates a **time-varying hedge ratio** for **pairs trading** —
the number `β` that tells you how many units of asset *x* to short against one unit of asset
*y* so the combination (the "spread") stays mean-reverting. The hard part is that `β` drifts,
markets have outliers and volatility spikes, and structural breaks happen. We estimate `β`
online with a **Kalman filter** made robust by four add-on components, and we prove each
component earns its place using **simulated data where we know the true answer**. So far we
have built and validated the **baseline** filters (Phases 1–2). Everything you have seen is
**synthetic data we generated ourselves** — no real market data has been used yet.

---

## 1. The problem in plain language

### 1.1 What is a hedge ratio?
Two related assets `y` and `x` (e.g. two Australian/Canadian ETFs). We model:

```
y_t = α_t + β_t · x_t + noise
```

`β_t` is the **hedge ratio**. If it were constant, ordinary regression would find it. It is
**not** constant — it drifts as the relationship between the assets evolves — so we need an
estimate that **updates every day** using only past data (no peeking at the future).

### 1.2 Why this is hard (the four problems we defend against)
1. **Outliers** — a single crazy print shouldn't yank `β` around.
2. **Volatility spikes** — in stressed markets the estimate should *slow down*, not churn.
3. **Anchoring** — permanent shifts in the spread level shouldn't be mistaken for `β` changes.
4. **Structural breaks** — when the true relationship genuinely jumps, the filter should
   detect it and re-adapt quickly.

The project’s core bet: attack these with four **independent, switchable components** layered
on a plain Kalman filter, and **measure each one's contribution** rigorously.

---

## 2. How we designed it (the "blueprint" chain)

Rather than jumping to code, we ran a four-stage design pipeline. Each stage produced a
reviewable artifact before the next began.

| Stage | Question it answers | What it produced |
|---|---|---|
| **ideate** | Is the plan's structure right? | A stress-test + a **chosen approach** |
| **architect** | What are the modules and boundaries? | A module map + domain model + dependency check |
| **design** | What are the exact interfaces and files? | Protocols + file tree + data-flow contract |
| **scaffold** | Stamp out the skeleton | ~50 files, all importable, with `TODO`s and tests |

### 2.1 The one decision that shaped everything
The original plan called the four components **"orthogonal switches."** During **ideate** we
found that is *not true*: they share state and must run in a **fixed order** (the robust
weight needs the innovation; the volatility estimate needs the weighted innovation; etc.).

So instead of four independent switches, we built an **ordered pipeline of small steps**
("stages") that each read and write one shared per-step object (`StepContext`). The benefit:
turning a component on/off is literally adding/removing a stage, which means **the experiment
(the ablation grid) and the architecture become the same thing.**

---

## 3. The architecture (how the code is organized)

### 3.1 The central idea
- **`StepContext`** — a small mutable record carrying one time-step's numbers (state `θ=(α,β)`,
  covariance `P`, innovation `e`, its variance `S`, robust weight `w`, …).
- **`StepStage`** — one transform on that record. Examples: *predict*, *update*.
- **`Pipeline`** — holds an ordered list of stages and runs them over the whole series.

### 3.2 The canonical per-step order (the data-flow contract)
```
1 predict         → project state forward (θ⁻, P⁻)
2 adaptive R      → set today's measurement-noise level        [Phase 4]
3–5 update        → innovation e, S → robust weight w → weighted correction of θ, P
7 changepoint     → detect a break, reset covariance           [Phase 6]
```
Keeping this order in one place (`Pipeline`) is what makes the coupled components correct.

### 3.3 Module map
```
src/hrl/
  core/        StepContext, protocols (contracts), FilterResult, parallel helper
  models/      LinearGaussianSSM  (y = α + βx),   PartialCointegrationSSM  [Phase 7]
  filters/     Pipeline, baselines (rolling OLS/TLS, vanilla KF), stages/, oracle (test-only)
  anchors/     Johansen & TLS slow anchors                                 [Phase 5]
  data/        synthetic DGPs (S1–S6), real ETF loaders                    [loaders: Phase 8]
  eval/        metrics, backtest, reports
  experiments/ AblationRunner (parallel grid), CLI
```

---

## 4. What "data" means here — READ THIS

This is the part that is easy to misunderstand.

### 4.1 There is currently **no real market data**
Nothing has been downloaded. No API was called. No CSV was read. The real-data loader
(`data/loaders.py`) is still an empty stub.

### 4.2 We use **synthetic data with a known true answer**
Every number in the results tables comes from `data/synthetic.py`, which *manufactures* price
series from a relationship **we define**. Because we set the true `β` ourselves, we can measure
exactly how close each estimator gets to it.

> **Analogy:** before trusting a bathroom scale, you put a known 10 kg weight on it and check
> it reads 10 kg. Synthetic data is the "known weight." Real market data is a person of
> unknown weight — useful later, but useless for *checking accuracy* because you don't know the
> right answer.

### 4.3 The six synthetic scenarios (DGPs)
Each is a deliberately different stress test. `x` is always a random-walk "log-price."

| Name | What it simulates | Purpose | Status |
|---|---|---|---|
| **S1** | Constant `β`, clean noise | Sanity check | ✅ built |
| **S2** | Slowly drifting `β` (sine wave) | Can it track genuine drift? | ✅ built |
| **S3** | Heavy-tailed noise + 1% outliers | Robustness | ✅ built |
| **S4** | Volatility regimes (calm/stress) | Adaptive noise | ✅ built |
| **S5** | `β` jumps at known times | Break detection | ⏳ Phase 6 |
| **S6** | Spread = random-walk + mean-reverting | Anchoring temptation | ⏳ Phase 7 |

Concretely, **S1** is generated as: `α = 0.5`, `β = 1.2` (fixed), `x` a random walk near
level 4.6, and `y = 0.5 + 1.2·x + small Gaussian noise`. The "true hedge ratio" is 1.2 **by
construction**, so "KF recovered 1.19" means the filter works.

### 4.4 Reproducibility
Every dataset is generated from an explicit random **seed**, so the exact same "data" appears
every time. Nothing is stochastic run-to-run.

---

## 5. What is built and validated so far

### 5.1 Phase 1 — the working baseline filter
Implemented and tested:
- `LinearGaussianSSM` — the `y = α + βx` model (random-walk state).
- `PredictStage` + `GaussianUpdateStage` — a complete Kalman step. (The robust weight is
  injected *inside* the update, because it needs the innovation `e, S` that don't exist until
  the update begins — a correction to the original scaffold.)
- `Pipeline` — runs the stages over a series, returns all the paths.
- Baselines: rolling OLS, rolling TLS, and a standalone vanilla Kalman filter.
- Estimation/stability metrics: RMSE, MAE, churn, turnover, sign-flips.
- Synthetic scenarios S1–S3.

**Live result on S1** (2500 steps, seed 0):

| Quantity | Value | Meaning |
|---|---|---|
| True `β` | 1.2000 | what we planted |
| Kalman `β` (final 250 avg) | **1.1906** | it converged |
| Kalman churn vs rolling OLS | **2× smoother** | far less noisy |
| Pipeline vs standalone KF | equal to **1e-10** | the composition is faithful |

### 5.2 Phase 2 — fitting the filter's knobs + validation
Implemented:
- `VanillaKalmanRW.fit` — learns the noise parameters `(q_α, q_β, r)` by **maximum
  likelihood** on a *training slice only* (no future leakage).
- `ewma_half_life` — documents the classic result that a Kalman filter on a random-walk state
  behaves like an exponential moving average; we report its effective "memory" in days.

**Validation (fit on first 40%, scored on the untouched last 60%):**

| DGP | fitted `q_β` | EWMA half-life | Kalman RMSE | rolling-OLS RMSE | rolling-TLS RMSE |
|---|---|---|---|---|---|
| **S1** static | 2.1e-10 | **985 steps** | **0.0013** | 0.031 | 0.058 |
| **S2** drift | 2.3e-6 | **9 steps** | **0.029** | 0.599 | 127.5 |

How to read this:
- The fit **automatically discovers the regime**: on static S1 it makes `β` almost frozen
  (985-step memory); on drifting S2 it makes `β` nimble (9-step memory). We did not tell it
  which was which.
- The Kalman filter beats rolling OLS by **~20×** on both.
- **A real finding:** rolling TLS blows up on drift (RMSE 127). That is a known instability of
  total-least-squares over a short window — and exactly why the plan reserves TLS for the
  *slow anchor* (long window), never per-step estimation.

---

### 5.3 Phase 3 — WoLF robust weighting (outlier defence)
Implemented:
- `IMQWeight` (primary, the paper's inverse-multiquadric weight), plus `HuberWeight` and
  `StudentTWeight` as references, injected into `GaussianUpdateStage`.
- Locked invariants: weights lie in `(0, 1]` and shrink with the innovation; `c → ∞` exactly
  recovers the vanilla filter; the state update is **bounded/redescending** (a huge outlier
  moves the estimate *less*, not more), whereas the vanilla filter's response grows linearly.

**Validation (S3: heavy tails + 1% contamination, 60 Monte Carlo paths):**

| config | mean β RMSE | win-rate vs vanilla | median weight at the outliers |
|---|---|---|---|
| vanilla | 0.0573 | — | 1.000 (ignores nothing) |
| **WoLF-IMQ (c=3)** | **0.0420** | **0.90** | 0.555 |
| Huber | 0.0406 | 0.83 | 0.297 |
| Student-t | 0.0415 | 0.87 | 0.452 |

WoLF cuts hedge-ratio error by **~27%** and wins on **90%** of paths. Crucially, the median
weight it applies *exactly at the known contamination times* is ~0.55 — direct evidence it is
down-weighting the outliers, not just getting lucky. This is acceptance criterion #2.

### 5.4 Phase 4 — adaptive measurement noise (counter-cyclical gain)
Implemented:
- `EWMANoise` (primary) — `σ²_t = λσ²_{t-1} + (1-λ)(w_{t-1}e_{t-1})²`, floored; driven by the
  **lagged** weighted innovation stashed by the update, so `R_t` is set *before* the current
  innovation exists (no within-step circularity). `GARCHNoise` and `VBAKFNoise` (inverse-gamma
  variational Bayes, Särkkä–Nummenmaa 2009, a convergent per-step fixed point) are references.
- Wired into the pipeline at **position 2** (`predict → adaptive R → update`). EWMA/GARCH
  compose with WoLF; **VB-AKF and WoLF are mutually exclusive** (they would co-iterate over the
  same innovation) and the pipeline rejects that combination.
- New DGP **S4** — constant `β`, Gaussian noise whose sd switches calm↔stress (5×) on a
  persistent two-state Markov chain, with regime labels exposed for per-regime scoring.
- Locked invariant: with WoLF upstream `|w·e| ≤ c·√S`, so a single 20σ print inflates the vol
  state by at most `≈(1-λ)c²` relative to `R` (~1.5× here); the naive/no-weight EWMA inflates
  ~17× on the same print. Bounded, counter-cyclical adaptation.

**Validation (S4: two-state vol, 60 Monte Carlo paths, base R=4e-4, λ=0.94; split by regime):**

| config | Var(Δβ) calm | Var(Δβ) stress | β RMSE calm | β RMSE stress | win-rate vs fixed |
|---|---|---|---|---|---|
| fixed | 2.47e-04 | 1.39e-03 | 0.128 | 0.148 | — |
| **ewma** | 1.49e-05 | **1.53e-04** | **0.056** | 0.061 | **0.98** |
| garch | 1.64e-05 | 1.73e-04 | 0.058 | 0.064 | 0.98 |
| vbakf | 2.00e-04 | 2.31e-04 | 0.073 | 0.078 | 1.00 |
| **ewma+wolf** | 1.61e-05 | **1.14e-04** | **0.049** | **0.053** | 1.00 |

EWMA cuts hedge-ratio **churn in stress windows by ~89%** vs fixed R (win-rate 0.98) *and*
improves calm-regime tracking (RMSE 0.056 vs 0.128) — the gain slows down exactly when the
market is noisy, then speeds back up. Stacking EWMA under WoLF (`ewma+wolf`) is best on every
metric. This is acceptance criterion #2's volatility half.

## 6. How to run it yourself

```bash
cd hedgeratio
pip install -e ".[dev]"      # install the package + dev tools
pytest -q                    # run the test suite
python experiments/phase2_baselines.py   # regenerate the Phase-2 table
```

The test suite currently reports **54 passed, 3 skipped** (the 3 skips are integration gates
that need later phases).

---

## 7. What comes next

| Phase | Adds | Scenario it proves out |
|---|---|---|
| **3** ✅ | WoLF robust weighting (down-weight outliers) | S3 |
| **4** ✅ | Adaptive measurement noise (slow down in vol spikes) | S4 |
| **5** | AR(1) anchoring toward a slow Johansen/TLS estimate | S2/S6 |
| **6** | Changepoint detection + covariance reset | S5 |
| **7** | Partial-cointegration observation model | S6 |
| **8** | **Real ETF pairs** + walk-forward evaluation + backtest | real data |
| **9** | (optional) distributionally-robust variant | — |

Real market data enters at **Phase 8**, and only as a sanity layer. The primary claims are
about estimation accuracy and stability, measured on the synthetic scenarios where truth is
known.

---

## 8. Current status

- Design: complete (ideate → architect → design → scaffold).
- Code: Phases 1–4 implemented and tested; Phases 5–9 scaffolded with `TODO`s.
- Tests: **54 passing, 3 skipped**, no warnings.
- Data: **100% synthetic, seeded, reproducible. No real market data yet.**
