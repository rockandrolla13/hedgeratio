# Fat-Tail Robustness Layer — Implementation & Statistical Test Specification

Companion to `hedge-ratio-filter-plan.md`. This document specs, at implementation level, how extreme fat tails are handled in the hedge-ratio filtering framework, and defines the statistical test suite that validates each mechanism. Fat tails enter through four channels; each has its own mechanism, its own module, and its own tests:

| # | Channel | Mechanism | Module |
|---|---------|-----------|--------|
| 1 | State update corrupted by extreme innovations | Bounded/redescending influence (WoLF-IMQ; Student-t filter) | filters/wolf.py, filters/student_t.py |
| 2 | Noise model corrupted by extreme innovations | Robustified innovation-vol recursion for R_t | filters/adaptive_r.py |
| 3 | Clustered extremes = structural break, not outliers | Changepoint layer fed with robust-weighted innovations | filters/changepoint.py |
| 4 | Entry/exit bands miscalibrated under non-Gaussian tails | Conformal (ACI) calibration of z-score bands | eval/conformal_bands.py (new) |
| 5 | Leverage points: extremes in the regressor, not the noise | Leverage-capped gain (diagnostic-gated) | filters/leverage.py (new) |

Conventions: e_t = y_t - H_t θ_pred (innovation), S_t = H_t P_pred H_tᵀ + R_t (predictive innovation variance), z_t = e_t / sqrt(S_t) (standardized innovation). All mechanisms are strictly causal. All MC experiments: 200 paths, T = 2,500, fixed seeds.

## 1. Mechanism 1 — Bounded-influence state update
### 1.1 WoLF-IMQ (primary)
w_t = (1 + e_t^2 / (c^2 * S_t))^(-1/2); K_t = w_t^2 P_pred H_tᵀ / (w_t^2 H_t P_pred H_tᵀ + R_t); θ = θ_pred + K_t e_t; P = (I - K_t H_t) P_pred.
Requirements: c=∞ reduces exactly to vanilla KF (rtol 1e-12). Default c on train split so w_t≥0.95 for |z_t|≤2.5; grid c∈{1,2,3,4,∞}. Redescending: for |e_t|→∞, w_t² e_t ≈ c² S_t/e_t → 0; update peaks at |e_t|=c·sqrt(S_t). Cross-validate vs github.com/gerdm/weighted-likelihood-filter (≤1e-8 over path).
### 1.2 Student-t measurement filter (efficiency benchmark)
Scale-mixture: eps_t~t_ν(0,R) ⇔ eps_t|λ_t~N(0,R/λ_t), λ_t~Gamma(ν/2,ν/2). Per-step fixed-point (3–10 iters, tol 1e-10): λ̂=(ν+1)/(ν+e_curᵀ R⁻¹ e_cur); R_eff=R/λ̂; KF update with R_eff; recompute e_cur. ν∈{3,4,5,8}, estimated on train by predictive MLE. Bounded but NOT redescending (§1.4-T3).
### 1.3 Huber gate (cheap reference)
Clip standardized innovation: e_t ← sign(e_t)·min(|e_t|, k·sqrt(S_t)), k=1.345. Reference only.
### 1.4 Statistical tests — Mechanism 1
T1 influence saturation (deterministic): sweep e∈[0,50√S]; vanilla linear (R²>0.999); WoLF max near c√S, strictly decreasing beyond 2c√S; t-filter monotone bounded.
T2 contamination sweep: π∈{0,0.5%,1%,2%,5%,10%}, outlier scale 100R. WoLF RMSE at π=5% ≤1.5× its π=0; vanilla degrades ≥5×.
T3 paired MC significance: per-path RMSE for {vanilla,WoLF,t,Huber}; Wilcoxon signed-rank WoLF vs vanilla (n=200; report p, Hodges–Lehmann). Holm–Bonferroni FWER 0.05.
T4 efficiency under null: on S1, RE=RMSE_robust²/RMSE_vanilla² ≤1.10 (WoLF default c), ≤1.05 (t, ν≥8).
T5 specification cross-over: clean t(3) → t-filter wins/ties; contaminated-Gaussian → WoLF wins. Documented decision rule.
T6 posterior covariance sanity: under S3, WoLF median S_t over outlier-adjacent windows (±5d) ≤1.2× unconditional median.
## 2. Mechanism 2 — Robust adaptive R_t
sigma2_t = lam sigma2_{t-1} + (1-lam)(w_{t-1} e_{t-1})²; R_t=max(sigma2_t,R_floor). lam∈{0.94,0.97}. Bound: |w_t e_t|≤c√S_t ⇒ single print inflates vol state by ≤≈(1-lam)c². GARCH(1,1) on weighted innovations (train only). VB-AKF (inverse-Gamma, forgetting ρ, 2–4 iters) as reference.
T7 single-outlier impulse (deterministic): one 20σ innovation. Robust: max_t R_t/R_true ≤1+2(1-lam)c²; naive ≈1+400(1-lam). Deafness duration: gain<50% steady-state ≤3d robust, ≥20 naive.
T8 counter-cyclical gain (S4): Var(Δβ|stress) adaptive < fixed, paired Wilcoxon; calm RMSE not worse >5% (non-inferiority, one-sided α=0.05).
T9 vol-forecast adequacy: Mincer–Zarnowitz regress e_t² on sigma2_t; HAC Wald; slope∈[0.7,1.3].
## 3. Mechanism 3 — Clustered extremes: break vs outlier
Detector consumes robust-weighted standardized z̃_t=w_t z_t. CUSUM default: C_t±=max(0,C_{t-1}± ± z̃_t - κ), κ=0.5; threshold h by sim (T10); on trigger P←P+P_reset, P_reset=diag(0,(0.25|β̄|)²), refractory 20d. BOCPD hazard 1/500 on z̃_t; reset when MAP run length<5.
T10 ARL₀ calibration: 500 S1 null paths; h so ARL₀≈500d; binomial test at α=0.05.
T11 masking (critical): S5 jump≥0.15; detection within 60d ≥90% WITH WoLF on. ROC as c varies.
T12 detector outlier-immunity: S3-contam no breaks; FA rate with weighted inputs ≤ T10 rate; raw z_t inflation reported.
T13 frontier vs inflated-Q: S5; methods {vanilla,Q×{2,5,10},CUSUM-reset,BOCPD-reset}; (reconv τ, sd(Δβ)); reset dominates Q×5,Q×10 (paired Wilcoxon, Holm).
T14 hard mode: S5+S3; composite RMSE within 25% of oracle (KF told true breaks+outliers). Headline stat.
## 4. Mechanism 4 — Conformal calibration of trading bands
Under t(3), 95% point |z|≈3.18 not 1.96. ACI (Gibbs–Candès 2021): s_t=|z_t|; α_{t+1}=α_t+γ(α-err_t), err_t=1{s_t>q̂_t}; q̂_t=(1-α_t) empirical quantile over trailing W_cal=250; γ∈{0.005,0.01}; warmup 250. Bands enter s_t>q̂_t(α_enter), exit s_t<q̂_t(α_exit). Filter-agnostic wrapper.
T15 unconditional coverage: Kupiec POF LR; conformal fail-to-reject on ≥90% cells; Gaussian-S bands mass-reject on S3/S4.
T16 conditional coverage: Christoffersen independence + joint LR on err_t; logit of err_t on lagged sigma2_t insignificant.
T17 PIT (diagnostic): Berkowitz LR + Anderson–Darling on filter PIT.
T18 Hill tail index (real pairs): Hill on |z_t| upper decile, bootstrap CI.
T19 economic materiality: backtest Gaussian vs conformal; Δtail-loss days, Δturnover, ΔSharpe with stationary-bootstrap CIs (Politis–Romano, block 20). Directional.
## 5. Mechanism 5 — Leverage points (extremes in x_t)
IMQ weight is function of e_t only; extreme x with ordinary innovation → outsized update via H_t=[1,x_t]. DGP S7: Δx_t~t(3) realistic daily vol, Gaussian eps, constant β; S7b simultaneous extreme Δx and eps. Leverage stat ℓ_t=H_t P_pred H_tᵀ/med_250(H P Hᵀ). Cap (if T20 bites): H_eff=H_t·min(1,sqrt(ℓ_max/ℓ_t)), ℓ_max≈9, gain only (innovation uses true H_t).
T20 leverage damage: S7/S7b paired Wilcoxon composite-no-cap vs vanilla vs composite-cap. Gate: cap in production only if uncapped S7 RMSE > S1 RMSE by >25% AND cap recovers ≥half the gap without hurting S1 (non-inferiority 5%).
## 6. Cross-cutting experimental protocol
Paired designs (common random numbers); Wilcoxon signed-rank on per-path metrics; Hodges–Lehmann effect sizes. Holm–Bonferroni FWER 0.05; primary hypotheses T3,T13,T15. Ablation interaction: 2⁴ factorial (WoLF,adaptive-R,anchor,changepoint) on S5+S3; linear model RMSE on main effects+pairwise interactions; harmful WoLF×changepoint (masking) or WoLF×adaptive-R (double-shrinkage) must be non-positive or <10% of largest main effect. Report: results/FAT_TAIL_REPORT.md auto-generated.
## 7. Test summary matrix
T1 M1 deterministic gate; T2 M1 MC gate; T3 M1 inferential gate(primary); T4 M1 ratio gate; T5 M1 report; T6 M1 gate; T7 M2 deterministic gate; T8 M2 non-inferiority gate; T9 M2 report; T10 M3 calibration gate; T11 M3 power gate(primary); T12 M3 binomial gate; T13 M3 inferential gate(primary); T14 M3 ratio gate(headline); T15 M4 inferential gate(primary); T16 M4 gate; T17 M4 report; T18 M4 report; T19 M4 partial; T20 M5 conditional.
## 8. References
Duran-Martin et al 2024 ICML PMLR235:12138 (WoLF); arXiv:2411.10153 (BONE); Roth et al 2017 arXiv:1703.02428 (Student-t); Gibbs & Candès 2021 NeurIPS (ACI); Kupiec 1995; Christoffersen 1998; Berkowitz 2001; Hill 1975; Politis & Romano 1994; Mincer & Zarnowitz 1969; Adams & MacKay 2007; Fearnhead & Liu 2007.
