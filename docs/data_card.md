# Data card — NASA C-MAPSS turbofan degradation

## Source

* Provider: NASA Prognostics Center of Excellence (PCoE).
* Data set: *Turbofan Engine Degradation Simulation Data Set*.
* Official download:
  `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
* C-MAPSS is simulated run-to-failure data for a turbofan engine model; it is not
  measured from physical equipment.

## Subsets used

| Subset | Operating conditions | Fault modes | Use here |
|--------|----------------------|-------------|----------|
| FD001  | 1                    | 1 (HPC degradation) | yes |
| FD003  | 1                    | 2 (HPC + fan degradation) | yes |
| FD002  | 6                    | 1 | not used |
| FD004  | 6                    | 2 | not used |

Only FD001 and FD003 are used, because the operational settings are near
constant there and the selective-evaluation questions are easier to isolate.

## File layout

```
train_{ds}.txt   run-to-failure trajectories of the training engines
test_{ds}.txt    trajectories truncated some time before failure
RUL_{ds}.txt     remaining cycles of the last recorded cycle of each test engine
```

Each text file is whitespace separated with 26 columns:

```
unit, cycle, op1, op2, op3, s1, s2, ..., s21
```

## Preprocessing

* 14 informative sensors retained:
  `s2 s3 s4 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21`
* Trailing windows of 30 cycles; windows shorter than 30 are front-padded by
  repeating the first row.
* 56 features per window: per-channel mean, per-channel standard deviation, last
  cycle value, least-squares slope over a centred time axis.
* Standardization uses per-channel mean and ddof-1 standard deviation estimated on
  the clean training engines of the relevant split only; channels with
  non-finite or practically zero standard deviation fall back to unit scaling and
  zero fault amplitude.
* Engine split: permutation of engines 1..100 by `numpy.random.default_rng(seed)`,
  then 70 train / 15 validation / 15 calibration. Validation engines are used for
  checks only, never for selection.

## Redistribution

The raw text files are **not** redistributed in this repository. Fetch them from
the official URL above and place them under `data/raw/` (see
`scripts/fetch_cmapss.md`).

## Known caveats

* Simulated data with a single degradation trajectory family.
* FD001 and FD003 have effectively constant operating conditions; results do not
  transfer to the variable-condition subsets without re-tuning.
* The test labels are provided per final cycle; truncation points inside the test
  trajectories use the raw `T - end_cycle` as truth, which assumes a linear
  degradation-to-failure mapping that the data set itself defines.
