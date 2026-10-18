# RiskSelRUL 0.1.1.dev0 API

This reference describes the `softwarex-dev` development branch. Its numerical interfaces preserve the frozen 0.1.0 computational baseline. The repository's main branch retains the earlier, different 1.0.0 implementation. Run `python -m pip install .` from this branch. NumPy is the only required runtime dependency for these interfaces.

## Common input conventions

Prediction, target and score inputs are finite one-dimensional numeric arrays of equal length. Smaller scores indicate preferred predictions. Invalid shapes, nonfinite values and mismatched lengths raise `ValueError`. Choose scores independently of evaluation labels. Targets measure error and the retrospective oracle; they do not choose the risk-ranked subset.

For correlated observations, these functions calculate descriptive sample metrics; they do not infer confidence intervals or automatically account for engine grouping. Fixed-count ranking uses the current cohort and is an offline diagnostic.

## Public metric interfaces

Import these four functions from `riskselrul`.

| Function | Inputs and behavior | Return |
| --- | --- | --- |
| `prediction_metrics(prediction, target, overestimate_margin=20.0)` | Computes prediction minus target errors. Margin must be finite and nonnegative. An overestimate is strictly greater than this margin. | Dictionary with `n`, `mse`, `rmse`, `mae`, `mean_error`, `overestimate_rate`. For an empty input, `n=0` and all error values are `None`. |
| `calibrate_threshold(calibration_scores, acceptance_rate=0.8)` | Nonempty finite score array; target rate in `(0,1]`. Uses NumPy's linear empirical quantile. Freeze this value before reading test outcomes and accept new scores `<= threshold`. | Float threshold. Realized test acceptance need not equal the calibration target. No distribution-shift guarantee is supplied. |
| `fixed_k_metrics(prediction, target, scores, k, sample_ids=None)` | Integer `k` from zero through cohort size, excluding booleans. Picks the `k` smallest scores. Optional IDs must match input length; ties use SHA-256 of string IDs, with input position as a final tie break. Supply stable unique IDs when reordering tied inputs. | Dictionary with `k`, `n`, `acceptance_rate`, `selected_indices`, `retained`, `oracle`, `full`, `random_expected_mse`, `selection_regret`. The three metric dictionaries follow `prediction_metrics`. For `k=0`, retained risks, random expectation and regret are `None`; selection is empty. |
| `risk_coverage_curve(prediction, target, scores, sample_ids=None)` | Same risk order and ID convention; evaluates each `k=1,...,n`. | List of dictionaries with `k`, `acceptance_rate`, `retained_mse`, `retained_rmse`, `oracle_mse`, `random_expected_mse`. Empty input returns `[]`; use `fixed_k_metrics(..., k=0)` to record full rejection explicitly. |

`random_expected_mse` is the analytical expected MSE under uniform sampling of a nonempty fixed-size subset, equal to full-cohort MSE. Its square root is **not** asserted to equal expected random RMSE. `selection_regret` is retained MSE minus oracle MSE, clipped below at zero for numerical roundoff. Oracle values use labels retrospectively and are not deployable scores. Returned `None` values serialize to JSON `null`.

## Executed minimal example

```python
from riskselrul import fixed_k_metrics

prediction = [10, 12, 9, 15, 8, 13, 11, 14]
target = [11, 10, 8, 14, 10, 12, 10, 11]
score = [.7, .1, .4, .8, .3, .2, .6, .5]
result = fixed_k_metrics(prediction, target, score, k=4)
print(result['selected_indices'])  # [1, 5, 4, 2]
print(result['retained']['mse'])   # 2.5
```

The release's companion `quickstart_arrays.py --out NEW_DIRECTORY` also saves `input.csv`, `curve.csv` and `result.json`, applies a separately calibrated threshold and verifies empty/all retention. Inputs are synthetic interface examples, not evidence of NASA prediction accuracy.

## Archived numeric inference

Import from `riskselrul.deploy`:

- `available_states()` lists the packaged numeric state inventory.
- `load_state(path_or_name)` reads an external NPZ path or a bundled state name, with `allow_pickle=False` and format checks.
- `predict(raw_windows, state)` returns one-dimensional `(prediction, risk_score)` arrays.
- `decide(raw_windows, state, condition='C2', acceptance_rate=0.8)` adds a boolean accepted array, using the single matching stored threshold. An unavailable condition/rate raises `ValueError`.

Windows must have shape `(n,30,21)` in physical sensor units. NaN is permitted and imputed using stored training statistics; infinity is rejected. All 78 states are bundled in the validated wheel and source archive. The frozen WP04A inference profile and the WP05A-derived `reproduce` training profile are distinct; the latter is not claimed to regenerate the former.

## Optional fixed benchmark

Install `python -m pip install '.[train]'` for `reproduce` and benchmark-table `audit`. Use `python -m riskselrul --help` and subcommand `--help` for exact arguments. `audit` expects the supplied benchmark's tables and configuration, rather than arbitrary user CSV schemas. It independently recalculates arithmetic without retraining a model. Installation/tests, arithmetic checks and raw-data model fitting are separate evidence scopes.


## Saved-table input validation

The auditor rejects header-only test tables and nonempty tables without `pred_` columns with `ValueError`. This also applies when an empty table appears alongside valid tables. Valid numerical evaluations preserve the 0.1.0 results. The three regression cases are included in `tests/test_audit.py`.
