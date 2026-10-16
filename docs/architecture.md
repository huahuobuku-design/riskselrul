# Architecture

## Data flow

```
C-MAPSS text (train_/test_/RUL_)
        |
        v
70/15/15 engine split  (data/cmapss.py: split, folds)
        |
        v
30-cycle trailing windows -> 56 features  (data/features.py)
        |
        +--> primary predictors: Ridge(alpha=100), HGB
        |    clean windows and clean+augmented windows  (models/predictors.py)
        |
        +--> out-of-fold residuals on training engines (5 folds x 10 truncations)
                 |
                 v
            error-risk heads: H_log, H_sq, pred_only, fault_indicator
            (models/risk_heads.py)
                 |
                 v
            calibration thresholds at coverage 0.8  (evaluation/pipeline.py)
                 |
                 v
            test streams (21 fault scenarios x 3 streams)
                 |
                 +--> full / threshold-accepted metrics
                 +--> fixed-k risk-ordered vs random vs oracle  (selective/decision.py)
                 +--> same-engine clean anchor: direction agreement, Spearman
                 +--> paired engine-cluster bootstrap  (evaluation/bootstrap.py)
                          |
                          v
                     per-sample tables + long metrics + bootstrap intervals
                          |
                          v
                     independent numeric audit  (evaluation/audit.py)
```

## Modules

| Module | Responsibility | Derived from |
|--------|----------------|--------------|
| `data/cmapss.py` | Load subsets, engine splits, folds, windowing | `data.py` |
| `data/features.py` | Sensor statistics and the 56-feature map | `data.py` |
| `faults/rng.py` | Explicit SHA-256 RNG keying | `rng_keys.py` |
| `faults/scenarios.py` | Fault scenarios, injection, imputation, indicator | `faults.py` |
| `models/predictors.py` | `fit_bundle`, `observe` | `train_entry.py` |
| `models/risk_heads.py` | `fit_heads`, `score` | `train_entry.py` |
| `selective/decision.py` | `order`, `cell` | `train_entry.py` |
| `selective/metrics.py` | `metrics`, `strata_metrics`, `anchor`, `flat` | `train_entry.py` |
| `evaluation/pipeline.py` | Stage 1 + stage 2 orchestration | `train_entry.py` |
| `evaluation/bootstrap.py` | Paired engine-cluster bootstrap | `bootstrap.py` |
| `evaluation/audit.py` | Standalone metric re-computation | `verify_metrics.py` |
| `deploy/lean.py` | Pure-NumPy inference from lean states | `lean_deploy.py` |

Consolidation changed interfaces, naming and comments only. The numerical
computations are intended to be identical to the source project; the regression
tests and the released per-sample tables are the check on that.

## Frozen environment

The released results were produced with:

```
python 3.12.14
numpy 2.5.4      pandas 3.0.6     scikit-learn 1.9.1
scipy 1.18.1     matplotlib 3.11.2    threadpoolctl 3.7.0    joblib 1.6.0
```

A single BLAS/OpenMP thread is enforced (`threadpoolctl.threadpool_limits(1)` and
the `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` variables) so that wall-clock timings
and floating-point reduction order stay comparable.

## Determinism

Every stochastic draw is addressed by a canonical key string

```
WP05A|v1|ds={ds}|seed={seed}|split={split}|purpose={purpose}|stream={stream}|engine={engine}|wend={wend}|fault={fault}
```

which is SHA-256 hashed and truncated to 64 bits to seed
`numpy.random.default_rng`. The `WP05A|v1|` namespace prefix is **frozen**: it
encodes the origin of the seeds, and changing it would invalidate every released
split, fault draw and result. Python's `hash()` is never used.

Label-free tie-breaking in orderings uses `SHA256(engine)` hex, then draw
position.

## Lean deployment states

A state is a numeric-only `.npz` (no pickled objects, no covariance, no bootstrap
ensemble) with keys:

```
indices, raw_MU21, raw_MU, raw_SD,
scaler_mean, scaler_scale, ridge_coef, ridge_intercept,
risk_mean, risk_scale, risk_coef, risk_intercept,
threshold_condition, threshold_coverage, threshold_value
```

`indices` selects the 14 informative channels out of the 21 columns; `raw_MU21`
imputes NaN; the `ridge_*` block produces the RUL and the `risk_*` block the risk
score. `threshold_*` stores the frozen calibration thresholds per condition and
coverage. See `deploy/states/states_index.csv` for the 78 released states and
`deploy/lean.py` for the reader.

## Relationship to the source project

This package is a consolidated extraction of a multi-round internal project
(`WP01` … `WP05j`). Process artefacts — coordination notes, per-round delivery
copies, review requests and failed attempts — are intentionally **not** part of
the software distribution; they remain in the source archive. The deviations
worth remembering are summarised in `docs/limitations.md`.
