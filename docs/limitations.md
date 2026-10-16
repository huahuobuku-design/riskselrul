# Limitations and negative results

This file collects everything the project found that did **not** support a
positive claim. It is part of the deliverable on purpose: the evaluation protocol
is only meaningful if the uncomfortable results stay visible.

## Scope of the evidence

* Results use two subsets (**FD001**, **FD003**) of a **simulation** data set,
  reused across every experiment round.
* The three training seeds share the same 100 test engines; they are training
  perturbations, not three independent external data sets.
* Nothing in this package supports generalisation to real equipment, to other
  C-MAPSS subsets, or to other fault regimes.

## Negative results

**N1 — No established novel scientific finding.**
The combination of a separately fitted error-risk head with quantile-based
rejection has direct antecedents (see *Antecedents* below). The package therefore
claims software value (reproducible baseline, honest protocol, lean deployment),
not algorithmic novelty.

**N2 — The primary predictor collapses under strong bias.**
On FD003, with the evaluation fault streams averaged over ten splits, the fraction
of engines whose (non-negative-clipped) prediction is exactly 0 rises from 4.1%
(clean) to 68.2% (bias_2sd) and 99.8% (bias_4sd), with full RMSE 17.98 → 78.14 →
85.93. Under these conditions absolute error is approximately the true RUL, which
is one reason a risk score can correlate with error without measuring anything
new.

**N3 — Rejection cannot restore clean error at 80% retention.**
For a fixed primary predictor, even a label-aware oracle that retains the 80
engines with the smallest true absolute error stays above the clean full-sample
RMSE (FD003: oracle 63.18/69.77/67.81 vs clean 17.98 for bias_2sd / bias_4sd /
drift_4sd). Changing the risk ordering or the threshold alone is therefore not
enough; the bound is descriptive and conditional on these fixed predictions.

**N4 — Label-capping counterexample.**
Removing the training-label cap (125 → raw RUL) *increases* the relative
selection gain while *worsening* the retained MSE, consistently across 12
dataset × model × seed cells (e.g. FD001 HGB: 5.04% → 39.04% gain but retained
MSE 170.20 → 309.58). Relative gain is therefore not comparable across complete
prediction systems.

**N5 — The FD003 jump needs no new ordering.**
With the primary predictor changed and the ordering held fixed, FD003 rises from
about 17% to about 64% relative gain; adding a new ordering contributes only
descriptively and does not restore the original absolute error.

**N6 — Lightweight heads degrade at equal k on FD003.**
At the same retained count, the 99-dim head is worse by +0.1666 MSE on average
(8/10 seeds) and the 57-dim head by +0.3175 (9/10 seeds) versus the 101-dim
reference. The 57-dim variant is *not* loss-free, only smaller and faster
(FD001 0.386 → 0.044 ms per window; state 39,168 → 3,456 bytes).

**N7 — Low coverage under strong bias.**
Under the C3_bias condition the accepted fraction falls to roughly 0.19–0.22 on
FD003, far below the nominal 0.8. High measured "gain" must not be used to hide
this coverage failure.

**N8 — Direction judgement is no better than a constant.**
With a common denominator (excluding only `|Δerror| <= 1e-10`), the log-residual
head beats the trivial "every fault increases error" reference in only 1 of 40
groups, by about 0.00076 percentage points. The head does not reliably tell
whether a given fault has hurt this engine.

**N9 — Key full texts were not obtained.**
RobustUQ and MSC-CAN official full texts and code were not retrievable; "not
read" is never treated as "no overlap". A published CQR-based RUL paper
(IJPHM 2026) was read and its public implementation inspected: the calibration
engines' labels enter supervised feature selection upstream, so the standard
split-conformal guarantee is not directly supported by that snapshot.

**N10 — Dependency on two reused simulation subsets.**
All coverage/error statements are conditional on FD001/FD003, the 70/15/15 split
rule, and the fixed fault scenarios implemented here.

## Recorded failures and corrections

* **WP05D numeric fix.** At `k = 100` the oracle-improvement space is 0 in exact
  arithmetic; summation order left residuals of ~1e-12…1e-10 and produced a
  meaningless non-NA ratio. The 100% case is now set explicitly to the full-sample
  error and the ratio reported as NA. The k=80 main results are unchanged.
* **WP04A first verify failure.** The first failure log was overwritten and only
  the author's written note plus the later successful log remain; the original
  error was **not** verified and no log was reconstructed.
* **Windows path bug.** The original validator split file names on `"/"` and
  failed on Windows before reaching any metric check. `evaluation/audit.py` uses
  `os.path.basename` so the audit runs on both platforms.
* **Numeric fallbacks.** A first correlation run hit a read-only pandas array;
  a supplementary check needed SciPy, which was absent in one environment and was
  replaced by an average-rank computation in pure NumPy. Neither change altered
  inputs or methods.

## Antecedents (why novelty is not claimed)

* NeurIPS 2020 — residual-squared regression risk plus empirical-distribution
  rejection on unlabelled samples.
* CSR 2026 — conformal rejection accounting for model-specific bias.
* NeurIPS 2024, Traub et al. — *Overcoming Common Flaws in the Evaluation of
  Selective Classification Systems*: base predictor and ordering must be
  evaluated jointly; method rankings can reverse.
* IJPHM 2026, Robinson — conformalised quantile regression for RUL intervals on
  C-MAPSS, used here as a lightweight point-prediction-plus-interval comparator.

The package reproduces the *engineering* of a selective RUL pipeline and the
*diagnostics* that reveal these limits; it does not claim the underlying ideas.
