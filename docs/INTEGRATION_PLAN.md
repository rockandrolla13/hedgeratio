# Post-Merge Integration Plan

Live working checklist for landing the two background agents (Phase 4 + Fat-Tail layer),
completing Phase 5 wiring, and standing up the 2⁴ ablation. Delete or archive once executed.

State at drafting: `master` @ `ac74b64` (Phases 1–3 + Phase-5 anchor groundwork).
Two agents running in isolated worktrees, branched from `8090f88`.

---

## Stage 0 — Discover branches & preflight

```bash
cd /media/ak/10E1026C4FA6006E/GitRepos/hedgeratio
git worktree list          # find the two agent worktrees + their branch names
git branch                 # confirm branch names (call them BR_P4 and BR_FT below)
git log --oneline BR_P4 -1 # sanity: each agent committed
git log --oneline BR_FT -1
git status                 # master must be clean before merging
```

Preflight gate: **do not merge a branch whose agent summary reported red tests or an
unfinished spec.** Read each agent's returned summary first; if a branch is incomplete,
re-dispatch or cherry-pick only the good files.

---

## Stage 1 — Merge order & rationale

Merge **Phase 4 first, Fat-Tail second.** Rationale: Phase 4 changes are more central
(`pipeline.py`, `adaptive_r.py`) and unblock the Fat-Tail pending tests; landing it first
means the one real conflict (`synthetic.py`) is resolved against a known-good base.

```bash
git merge --no-ff BR_P4    # expect: clean except possibly synthetic.py dataclass
# resolve, test, commit
git merge --no-ff BR_FT    # expect: synthetic.py conflict (S7 vs S4/regime)
# resolve, test, commit
```

Use `--no-ff` so each agent's work is a reviewable merge commit preserving authorship.

---

## Stage 2 — Expected conflicts & resolution (file-by-file)

| File | P4 touches | FT touches | Conflict? | Resolution |
|---|---|---|---|---|
| `data/synthetic.py` | `s4_stochvol` body, `regime` dataclass field | `s7_leverage` fn, `_REGISTRY += S7`, maybe a field | **YES** | Union: keep both functions, both registry entries, both dataclass fields; ensure module constants (`_ALPHA` etc.) not duplicated |
| `filters/pipeline.py` | noise wiring in `from_config`, VB↔WoLF guard | — (forbidden) | No | Take P4 |
| `filters/stages/adaptive_r.py` | full impl | — (forbidden) | No | Take P4 |
| `config.py` | (should be none) | — | No | If P4 added fields, keep |
| `eval/__init__.py` | — | exports conformal/stats | No | Take FT |
| `eval/*.py` (new) | — | stats, conformal_bands, fat_tail_report | No | Take FT |
| `filters/student_t.py`, `filters/leverage.py` | — | new | No | Take FT |
| `docs/WALKTHROUGH.md` | Phase-4 section | — (forbidden) | No | Take P4 |
| `docs/FAT_TAIL_SPEC.md` | — | new | No | Take FT |
| `anchors/*`, `tests/unit/test_anchors.py` | — | — | No | Already on master |
| `tests/unit/test_*` | `test_adaptive_r.py` | `test_fat_tail_*.py` | No (disjoint names) | Take both |

**`synthetic.py` merge recipe** (the only manual one):
1. Keep the P4 version of `SyntheticSample` and add any FT-added optional field to it (union of fields).
2. Keep both `s4_stochvol` (+ helpers) and `s7_leverage`/`s7b` function bodies.
3. `_REGISTRY` must contain S1–S6 (existing) **and** S7[/S7b]. Verify with a post-merge test.
4. Confirm no duplicate `_x_series`/constants.

---

## Stage 3 — Post-merge verification gate

```bash
pip install -e ".[dev]" -q         # reinstall so the editable install matches merged src
python -m pytest -q                 # FULL suite must be green
python -c "from hrl.data.synthetic import _REGISTRY; print(sorted(_REGISTRY))"  # S1..S7 present
python -c "import hrl.eval.conformal_bands, hrl.filters.student_t, hrl.filters.leverage"  # FT imports
python experiments/phase4_adaptive_r.py   # smoke the P4 experiment
```

Gate: all prior tests + both agents' new tests pass; 0 import errors; expected skip count =
old 3 (WoLF-ref, oracle, walkforward) **minus** any the agents turned real **plus** FT's
`test_fat_tail_pending.py` skips (T7–T14). Record the new pass/skip counts.

Commit the merge resolution before moving on.

---

## Stage 4 — Complete Phase 5 (anchor wiring)  [depends on: merge done]

The anchor *estimators* are already on master; this wires them into the filter.

1. **History access for the anchored stage.** In `Pipeline.run`, stash the full arrays once
   per run into the context: `ctx.extra["_y"] = y; ctx.extra["_x"] = x` (set before the loop).
   The anchored stage reads `anchor.anchor(ctx.t, ctx.extra["_y"], ctx.extra["_x"])`; the
   anchor slices `[:t]` internally (already causal). Minimal, keeps the `StepStage(ctx, model)`
   signature intact.
2. **`filters/stages/anchored.py`** — implement `AnchoredTransitionStage.apply`:
   `β̄ = anchor.anchor(...)`; if `β̄` is nan (pre-`min_obs`) fall back to plain RW predict;
   else `θ⁻_β = β̄ + φ(θ_β − β̄)`, `θ⁻_α = θ_α` (RW), `P⁻ = F P Fᵀ + Q` with
   `Q_β = q_beta` (stationary var `q_beta/(1−φ²)`). Keep `name = "predict"` (occupies the
   single transition slot).
3. **`Pipeline.from_config`** — replace the anchor `NotImplementedError`: when `cfg.anchor !=
   "none"`, use `AnchoredTransitionStage(make_anchor(cfg.anchor, cfg), phi=cfg.phi,
   q_beta=cfg.q_beta)` as the transition stage instead of `PredictStage`. Add a
   `make_anchor(kind, cfg)` factory (`johansen`/`tls`) mirroring `make_weight_fn`.
4. **S6 DGP** — implement `s6_pci` in `synthetic.py` (spread = RW + AR(1)(ρ≈0.9)); it is the
   anchoring temptation case.
5. **Tests** — real `test_anchored.py`: on S2 (drift) anchored RMSE within tolerance of RW-KF;
   on S6 anchored β-variance materially below RW-KF while filtered-spread ADF holds/improves;
   **and the anchor↔changepoint interaction test** flagged in ideate (add once Phase 6 lands).
6. **Experiment** `experiments/phase5_anchor.py` → `results/phase5_anchor.md` (S2/S6).

---

## Stage 5 — The 2⁴ ablation runner  [depends on: Phases 4 done; 6 for full grid]

1. **`experiments/runner.py`** — implement `AblationRunner.expand()` to build the factorial
   over the available axes:
   - `weight_fn ∈ {none, imq}` (Phase 3 ✅)
   - `noise_model ∈ {fixed, ewma}` (Phase 4 ✅ after merge)
   - `anchor ∈ {none, johansen}` (Phase 5 ✅ after Stage 4)
   - `detector ∈ {none, cusum}` (**Phase 6 — not yet**)
   Until Phase 6 lands, run the **2³** grid (drop the detector axis) and `log()` that the
   4th axis is deferred (no silent truncation). Cross with DGPs {S3, S5-hard, …} × N paths.
2. **`run_cell(cell)`** — top-level picklable fn: `generate/​load` sample → `Pipeline.from_config`
   → `run` → score via `eval.metrics` → return a flat dict (config flags + metrics). Runs under
   `core.parallel.parallel_map`.
3. **Interaction analysis** (spec §6): fit a linear model of per-path RMSE on main effects +
   pairwise interactions on the S5+S3 hard-mode grid. Flag any **positive** WoLF×changepoint
   (masking) or WoLF×adaptive-R (double-shrinkage) interaction ≥10% of the largest main
   effect. Put this in `experiments/ablation_interactions.py`.

---

## Stage 6 — Unblock the Fat-Tail pending tests  [depends on: Phases 4/6]

The agent left `tests/unit/test_fat_tail_pending.py` as documented skips. After merge:
- **Now implementable (Phase 4 present):** T7 (R_t impulse response), T8 (counter-cyclical
  gain — may overlap P4's own S4 test; keep the T8 form with Wilcoxon + non-inferiority),
  T9 (Mincer–Zarnowitz). Convert these skips to real tests against `adaptive_r`.
- **Still pending (need Phase 6 changepoint):** T10–T14 — keep skipped until the detector
  lands, then implement (T11 masking and T13 frontier are primary gates).
- **Conformal into backtest:** wire `eval/conformal_bands.aci_bands` into `eval/backtest.py`
  (currently a stub) so T19 (economic materiality: Gaussian vs conformal bands) can run; this
  is Phase-8 work but the plumbing can land now.

---

## Stage 7 — Reports

- `eval/reports.py::write_report` → `results/REPORT.md` (main ablation tables + blame plot).
- `eval/fat_tail_report.py::write_fat_tail_report` → `results/FAT_TAIL_REPORT.md` (per-mechanism
  test tables; fill mechanisms 1/4/5 now, 2/3 after Phases 4/6). Both are gitignored outputs.

---

## Stage 8 — Land it

```bash
python -m pytest -q            # full green
git add -A && git commit -m "Integrate Phase 4 + Fat-Tail layer; wire Phase 5 anchors; 2^3 ablation"
git push origin master
gh repo view rockandrolla13/hedgeratio --web   # eyeball CI-less sanity
```

Update `docs/WALKTHROUGH.md` status block (phases done, test counts) as the final edit.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Agent branch red/incomplete | Stage-0 preflight reads summaries; cherry-pick good files if needed |
| `synthetic.py` merge drops a DGP | Stage-3 asserts `_REGISTRY == {S1..S7}` |
| Anchored stage lookahead bug | Anchors already unit-tested for causality; Stage-4 reuses them unchanged |
| VB-AKF + WoLF composed by accident in ablation | P4's `from_config` guard raises; ablation skips that cell |
| WoLF masks breaks (interaction) | Stage-5 interaction model + (post-Phase-6) T11 masking gate |
| Editable install points at stale src after merge | Stage-3 reinstalls `-e` before testing |
