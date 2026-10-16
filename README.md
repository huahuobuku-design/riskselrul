# riskselrul

Risk-aware **selective remaining-useful-life (RUL) prediction** on the NASA
C-MAPSS turbofan degradation data set.

The toolkit trains a primary RUL predictor, fits a **label-free error-risk head**
on out-of-fold residuals, and turns the risk score into an **accept / reject**
decision using a threshold fitted on an independent calibration stream. Alongside
the pipeline it ships an evaluation protocol that keeps the uncomfortable results
visible: the coverage / error trade-off, fixed-`k` ranking diagnostics against
random selection, label-aware oracle lower bounds, a paired engine-cluster
bootstrap, and an independent numeric audit that imports none of the experiment
modules.

> **What this software is not.** The error-risk-head-plus-quantile-rejection
> design has direct antecedents in the literature (see
> [`docs/limitations.md`](docs/limitations.md)). This package does not claim
> algorithmic novelty. It provides a reproducible implementation and an honest
> evaluation protocol for selective RUL prediction.

## Installation

```bash
git clone https://github.com/huahuobuku-design/riskselrul.git
cd riskselrul
python -m pip install -e .
```

Requires Python >= 3.10. Dependencies are listed in `requirements.txt`; the
frozen environment used for the released results is recorded in
`docs/architecture.md`.

## Data

C-MAPSS is **not redistributed** in this repository. Download the turbofan
degradation data set from NASA PCoE and place the per-subset text files under
`data/raw/` as

```
data/raw/train_FD001.txt  data/raw/test_FD001.txt  data/raw/RUL_FD001.txt
data/raw/train_FD003.txt  data/raw/test_FD003.txt  data/raw/RUL_FD003.txt
```

See [`scripts/fetch_cmapss.md`](scripts/fetch_cmapss.md) for the official URL and
the expected column layout. The data root can also be set with the
`RISKSELRUL_DATA_ROOT` environment variable or `--data-root`.

## Quickstart

```python
from riskselrul.deploy import lean
import numpy as np

state = lean.load("deploy/states/FD003_seed42_all_linear_57.npz")
raw = np.load("windows.npy")            # shape (n, 30, 21), NaN allowed
pred_rul, risk_score, accept = lean.decide(raw, state, condition="C2", coverage=0.8)
```

Full training and evaluation:

```bash
riskselrul train --datasets FD001 FD003 --seeds 42 43 44 --out runs/
riskselrul audit --run runs/
```

Or run the shipped examples:

```bash
python examples/01_train_and_evaluate.py
python examples/02_coverage_error_curve.py
python examples/03_load_lean_state.py
```

## What is in the box

| Path | Contents |
|------|----------|
| `src/riskselrul/data/` | C-MAPSS loading, engine splits, folds, windowing, features |
| `src/riskselrul/faults/` | Sensor-fault scenarios and deterministic RNG addressing |
| `src/riskselrul/models/` | Primary predictors (Ridge, HGB) and error-risk heads |
| `src/riskselrul/selective/` | Risk ordering, decision rules and metrics |
| `src/riskselrul/evaluation/` | Training pipeline, paired bootstrap, independent audit |
| `src/riskselrul/deploy/` | Pure-NumPy inference from lean deployment states |
| `deploy/states/` | 78 released lean states (57/99-dim linear heads, FD001/FD003) |
| `reproducibility/` | Freeze hashes, numeric audits and reproduction comparisons |
| `docs/` | Architecture, data card, limitations |
| `tests/` | Regression tests (lean forward, metric identities, fault generation) |

## Evaluation protocol

* **Selective prediction.** Accept a prediction when its risk score is at or
  below a threshold fitted on a calibration stream; the target coverage defaults
  to 0.8.
* **Fixed-`k` diagnostics.** For a preset retained count, compare the risk-ordered
  subset against an equally sized random selection *and* against a label-aware
  oracle over the same fixed predictions. The oracle is a diagnostic lower bound,
  not a deployable method.
* **Averaging.** Metrics are averaged with equal weight over (seed x stream)
  combinations, never over predictions, scores or labels.
* **Tie-breaking.** Label-free: by risk score, then engine-hash, then draw
  position.

## Reproducibility

* `reproducibility/FROZEN_HASHES_wp04a.txt` - frozen input/output hashes.
* `reproducibility/numeric_audit_*.json` - independent numeric audits.
* `src/riskselrul/evaluation/audit.py` - the standalone auditor.
* `scripts/reproduce_all.sh`, `scripts/reproduce_all.ps1` - one-shot re-runs.

RNG addressing is explicit and platform independent: every draw is keyed by
`dataset|seed|split|purpose|stream|engine|window_end|fault` and hashed with
SHA-256. See `src/riskselrul/faults/rng.py`.

## Limitations

The honest results, including everything that did **not** work, are collected in
[`docs/limitations.md`](docs/limitations.md). Read it before citing any number
from this package.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff).
