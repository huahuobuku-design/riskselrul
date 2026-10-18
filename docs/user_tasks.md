# Three practical diagnostic tasks

These are self-designed functional scenarios with synthetic inputs, not reports
of independent users or empirical prediction performance. Install the NumPy-only
package, then run `python examples/05_user_tasks.py --out user_tasks_output`.
Choose a new output folder on each run. Inspect `input.csv` and `result.json`.

1. **Compare two selectors at the same workload.** The supplied four predictions
   and targets are held fixed. At k=2, selector A retains rows a/c with MSE 0.5;
   selector B retains b/d with MSE 10.0. This answers which ordering works better
   for this cohort at the same retained count.
2. **Check whether a frozen threshold still serves a shifted cohort.** A threshold
   of 0.25 is calibrated at target acceptance 0.5. Shifted scores 0.6–0.9 lead to
   zero accepted predictions. The saved MSE is JSON null, not zero.
3. **Distinguish ranking from operational acceptance.** Fixed k=2 on that same
   shifted cohort still retains half the rows with MSE 2.0. This offline result
   does not repair the frozen threshold's zero acceptance.

To use your own model, replace predictions and scores with its outputs and keep
evaluation labels separate until scoring. Do not choose scores or thresholds by
optimizing against evaluation labels. The arrays can represent any regression
task; the optional raw-data command remains a fixed C-MAPSS benchmark.

## Fresh raw-data path

With `.[train]` installed, `python scripts/cold_start.py --dataset FD001 --out
NEW_DIRECTORY` downloads the NASA-hosted archive, checks the selected dataset
against stored hashes, trains seed 42 / stream 9001 with 2,000 bootstrap draws,
and audits its output. It saves the URL, archive digest, command exit codes,
timings and complete training records. Use FD003 for the other supported subset.
This downloads raw data for your local use; raw data are excluded from CI artifacts.
