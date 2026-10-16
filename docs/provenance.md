# Provenance and migration

The project archive is preserved outside this package. `reproducibility/source_manifest.json` lists the relative archive path and source/destination SHA-256 for every migrated file and copied state.

## WP05A raw-data reproduction

Five numerical source modules and the frozen configuration were migrated from the reviewed WP05A bundle. Numerical model/fault/RNG/feature/bootstrap formulas are preserved. Changes are package-relative imports, a packaged config path, ordinary dataset/seed/stream arguments, removal of the old conversational phase-authorization flag, and an unsuccessful exit code for an incomplete bootstrap.

The original `anchor.csv` is a historical conditional diagnostic that omits zero score changes. It is retained for traceability. `direction_common_denominator.json` from the independent audit applies a shared denominator and counts unchanged risk as a direction failure when error changes. Do not compare the old conditional percentage to a constant baseline with a different denominator.

## WP04A inference

The 78 NPZ states are copied byte-for-byte from the independently reviewed local WP04A `deploy` directory. The public inference functions are a readable implementation of the archived NumPy forward path, with shape/scale/finite-value checks added. State parity is checked separately for 57 and 99 features and for missing/stuck/drifting inputs.

WP04A and WP05A are separate experimental profiles. Their predictors, risk heads, and sampling rules differ. The packaged training command does not reconstruct the archived WP04A states. Source-level parity and measured timing are also separate: this package has not inherited an automatic 4.4x speed claim from the old hardware/window sample.

## Excluded materials

Private coordination files, access credentials, virtual environments, original data, repeated delivery archives, third-party article PDFs and third-party code copies are excluded. The code here originates from this project's generated/reviewed implementation, not from the external RiskAware-CQR repository examined in WP05J.
