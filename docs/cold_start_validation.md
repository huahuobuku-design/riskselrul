# Fresh execution verification 2026-10-18

Tested source: 5a90e3140d8d9c0610b7e8f6a7121d0f6277fb32. [Actions run 39124409463](https://github.com/huahuobuku-design/riskselrul/actions/runs/39124409463).

All six jobs succeeded: four installed-wheel combinations (Ubuntu/Windows,
Python 3.10/3.12), each passing 21 tests and executable examples; two fresh
Ubuntu raw-data jobs (FD001/FD003). Local Windows/Python 3.12.14 checks also
passed 21 tests and all examples. Each dataset then completed the full local
download, strict hash check, model fitting and audit path with seed 42, stream
9001 and 2,000 bootstrap draws. No original NASA data are redistributed.

Each of the four raw runs passed 5,392 saved-table checks. Further independent
review per run rebuilt 1,024 scalar metrics, inspected 5,600 OOF fitting
exclusions and six fold/final sensor normalization records. S3 preserves outputs
and a portable checker. Full three-seed/three-stream expansion remains in S2.

Fresh-download testing identified the NASA outer ZIP wrapper. The helper now
copies its single CMAPSSData.zip member before invoking the existing safe data
extractor. No predictor or selector calculation changed.

## Re-running and numerical scope

Use `python scripts/cold_start.py --dataset FD001 --out NEW_FOLDER` after
installing the train extra. Repeat with FD003 for the other supported subset.

Both platforms execute the full workflow and independently reproduce their
saved arithmetic. Refit outputs are not equivalent across platforms. Ridge
point predictions differ by at most 1.61e-11, while augmented HGB predictions
differ by up to 8.96 cycles (FD001) and 10.17 cycles (FD003). Across the 336
predictor/head/scenario cells per dataset, maximum accepted-count differences
are ten and nine engines; retained RMSE differences reach 2.98 and 4.22 cycles.
These observations should not be described merely as bitwise roundoff. No
cause has been established in this finalization pass. Use frozen S2 outputs
for the exact article values and retain version/environment records for new fits.

See reproducibility/cold_start_execution_review.json for full differences and
reproducibility/cold_start_remote_ci_evidence.json for CI log excerpts/artifact hashes.
