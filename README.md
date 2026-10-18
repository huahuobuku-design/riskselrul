# RiskSelRUL

A Python software toolkit for reproducible sensor-fault experiments and selective remaining useful life (RUL) diagnostics on NASA C-MAPSS FD001/FD003.

It connects raw data checks, engine-grouped out-of-fold error estimation, calibration, held-out fault evaluation, paired engine bootstrap, and independent arithmetic auditing. A separate NumPy-only interface runs 78 archived WP04A numeric inference states.

Use the NumPy interfaces with predictions and scores from any regression model, or reproduce the supplied sensor-fault workflow on C-MAPSS simulated engine trajectories.

## Paper and reproducibility materials

The [SoftwareX manuscript draft](paper/manuscript_v0.5.md), [Word manuscript](paper/RiskSelRUL_SoftwareX_v0.5.docx), [API reference](docs/API.md), and [supplementary materials](paper/README.md) accompany this version. The v0.5 manuscript is the reviewed local draft prepared before this repository update; its permanent-link placeholders are pending editorial revision.

## Install

Python 3.10 or newer:

```sh
python -m pip install .              # NumPy-only metrics and archived inference
python -m pip install '.[train]'      # Raw-data experiments and arithmetic audit
python -m riskselrul --help
```

The package must be installed before running the examples. Training dependencies are optional and are not imported by the metrics or inference interfaces.

## Five-minute array example (no NASA download)

This development branch contains **0.1.1.dev0**. It adds saved-table input guards to the frozen 0.1.0 benchmark used for the supplementary experiments. Clone this branch with `git clone --branch softwarex-dev https://github.com/huahuobuku-design/riskselrul.git`, then install from its root; the earlier public validated-v0.1.0 tag remains available as the experiment baseline.

```sh
python -m pip install .
python examples/quickstart_arrays.py --out example_output
```

The synthetic input produces `input.csv`, `curve.csv` and `result.json`. Fixed k=4 retains four of eight inputs with MSE 2.5; a separately calibrated threshold retains three with MSE 3.0. These values demonstrate the API and are not NASA model results. Choose a new output directory each time.

The core requires NumPy. Training and saved benchmark-table auditing use `python -m pip install '.[train]'`. All 78 numeric archived inference states are included in the wheel and source distribution. Source tests use pytest (`python -m pip install '.[train,dev]'`, then `python -m pytest`); the different 1.0.0 implementation uses unittest.

## General regression reuse

With the training extra installed, run `python examples/04_generic_regression.py --out generic_example`.
It exports reproducible OOF/calibration/test results for the diabetes and synthetic
Friedman examples. See [the example guide](docs/extended_examples.md).

## Optional raw-data benchmark

Obtain the original C-MAPSS ZIP from NASA as described in [the data card](docs/data_card.md). Raw data are not redistributed in the package.

```sh
riskselrul extract-data --zip /path/to/CMAPSSData.zip --out data/raw
riskselrul inspect-data --data-root data/raw --dataset FD001 --strict-hashes
riskselrul reproduce --data-root data/raw --dataset FD001 --seed 42 --streams 9001 --out runs/fd001-s42
riskselrul audit --run runs/fd001-s42
riskselrul curve --table runs/fd001-s42/test_FD001_s42_st9001.csv --prediction pred_ridge_clean --risk score_ridge_clean_H_log --out runs/fd001-s42/clean_curve.csv
```

Paths work on Windows and POSIX. Choose a new output directory for every run. Training logs, exit codes, fit boundaries, split IDs, configuration/data hashes, OOF rows, calibration thresholds, endpoint predictions, bootstrap intervals, and audit results are saved there.

The `reproduce` command fits all four fixed predictors (Ridge/HGB, clean/augmented), all four risk scores, five grouped OOF folds, and 21 test scenarios. It performs the entire raw-data chain for the requested dataset/seed/streams. It does not reuse OOF caches. The default bootstrap has 2,000 replicates; `--bootstrap-reps 20` is a faster pipeline check and must not be cited as the full analysis. To reproduce the archived full experiment, run both datasets, seeds 42/43/44 and streams 9001/9002/9003. Each command represents one training seed; results from different seeds are not silently pooled.

## NumPy-only archived inference

```python
from riskselrul.deploy import load_state, decide

state = load_state("FD001_seed42_all_linear_57.npz")
# raw_windows: numeric array (n, 30, 21); physical sensor values, with NaN allowed.
predicted_rul, risk_score, accepted = decide(raw_windows, state, "C2", 0.8)
```

Use `riskselrul list-states` to inspect the available states. State files contain numeric/string arrays only and are loaded with `allow_pickle=False`. Conditions are fixed calibration protocols, not detected fault labels: choose a protocol in advance. Realized acceptance may differ sharply from its calibration target.

**These states belong to the archived WP04A profile, not to the migrated WP05A training run.** The two profiles differ in upstream predictors, risk features, and training protocol. Version 0.1 provides archived inference but does not claim that `reproduce` regenerates the WP04A states. Their separate lineage is documented in [provenance](docs/provenance.md).

## Reusable diagnostics

```python
from riskselrul import fixed_k_metrics

result = fixed_k_metrics(prediction, target, risk_score, k=80, sample_ids=engine_ids)
```

Selection uses risk and label-free tie breaking. Targets enter error measurement and the retrospective oracle only. Zero accepted samples produce `None` risks, never a zero-error success. Uniform random retention is represented by its **expected MSE**, not by a misleading expected-RMSE approximation.

Fixed-k curves and oracle rankings are offline diagnostics. They are not deployable decision rules. The deployable form uses a threshold calibrated before observing test outcomes.

## Verification and examples

```sh
python -m pip install '.[train,dev]'
python -m pytest
python examples/03_archived_state.py
```

See [validation](docs/validation.md), [architecture](docs/architecture.md), and [limitations](docs/limitations.md). Validation records distinguish synthetic tests, saved-result arithmetic, legacy state parity, and a fresh raw-data reproduction. Cross-platform bitwise identity is not promised for fitted tree risk scores.

## License and release status

This source uses Apache-2.0. Human authorship/contact details and a persistent archive DOI remain unfilled; see the [release checklist](docs/release_checklist.md). A Git tag is not a DOI archive. No DOI, journal acceptance, external users, or speed measurements are invented. This directory contains no access token, private coordination database, chat screenshot, third-party paper source code, or downloaded article PDF.


The archived `reproducibility/source_manifest.json` describes the original 0.1.0 review input. The current development source inventory is `reproducibility/candidate_source_manifest.json`.
