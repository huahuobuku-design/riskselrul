# Validation scopes

Executed local validation is described below. Portable summaries of subsequent CI and AI-operated Debian verification are included in `reproducibility/ci_verification.json` and `reproducibility/grok_numeric_review.json`; full private coordination and machine-identifying logs are excluded.

Tests cover:

1. Selection is independent of target labels, deterministic score ties, all/zero retention, oracle lower bounds, and the distinction between expected random MSE and RMSE.
2. Strict engine/cycle/data shape checks and unsafe ZIP rejection.
3. All packaged archived states load without object deserialization, and forward parity against the archived implementation on clean and corrupted numerical windows.
4. Fresh raw-data training with engine OOF boundaries, independent recalculation, and comparison with archived per-engine predictions for the selected profile.
5. Wheel installation and CLI execution outside the source directory.

Synthetic numerical fixtures validate implementation behavior only. They are never reported as NASA performance. Saved-table audits do not execute training. A fresh training run is identified separately by dataset, seed, streams, bootstrap repetitions, logs, and actual exit code.

The portable frozen-state tests use rtol=1e-11 and atol=1e-9 for floating outputs and require identical acceptance flags. The local validation additionally reports whether outputs were bitwise identical; that observation is not a cross-platform promise.

The core installation excludes training libraries. Core examples and tests must run under that dependency boundary. The trained model/profile need not be bitwise identical across all platforms; deviations in predictions, scores, and accepted sets are reported separately.

## Historical 0.1.0 raw-data migration check (2026-10-17)

On Windows/Python 3.12.14, the migrated FD001/seed42/stream9001 command was run from the archived original NASA files with 2,000 bootstrap replicates. The training child reported 80 seconds and completed. Independent numeric audit: 5,392 checks, zero mismatches. This run covers 2,100 terminal engine/scenario rows and all four point predictors.

Against the prior independent WP05A local reproduction, 8,400 point predictions, 33,600 risk scores, 33,600 acceptance flags, and 33,600 fixed-k flags were exactly equal. All 5,600 OOF rows excluded their own engine from the recorded fitting set. Clean sensor means/standard deviations were independently recalculated for the five folds and final fit; their maximum difference was zero.

An initial package run completed model training but failed in the new auditor because an archived underscore column name was read as a dotted name. The auditor was corrected, a regression test was added, and a second fresh raw-data run passed. The first run/logs remain in the local validation archive. At that initial migration stage the full two-dataset/three-seed/three-stream analysis had not been re-executed. The later expansion is described below.

These are migration/integration results, not new evidence of algorithm superiority. The compact real example table is in `reproducibility/example_metrics_FD001_s42_st9001.csv`; it is not a synthetic benchmark and is not the complete multi-seed analysis.


## Expanded 0.1.0 experiments and 0.1.1.dev0 software checks

The 2026-10-17 expansion executed six complete dataset/seed fits and eighteen
dataset/seed/fault-stream cases on FD001/FD003. Received output tables, fitting
boundaries, command logs and conditional interval files accompany S2.
Independent local review recomputed all point metrics, reconstructed fold
preprocessing from the original raw files, refitted six non-RUL examples and
reconstructed all 2,688 representative interval rows by explicit resampling.
Results matched within numerical tolerances. The current review did not refit
C-MAPSS predictors. See `reproducibility/deep_review.json` and the Chinese review.

The 0.1.1.dev0 candidate adds audit schema/empty-table guards and an executable
general-regression example. It passed 21 local installed-wheel tests and both
quickstart examples. The earlier candidate CI built and installed a wheel and ran tests and examples on Ubuntu and Windows/Python 3.12, without raw training. The new finalization workflow adds Python 3.10 and separate fresh-download raw-training jobs; its results are recorded separately after actual execution.
