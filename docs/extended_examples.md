# General regression example

Install the source package with `python -m pip install '.[train]'`, then run:

```sh
python examples/04_generic_regression.py --out generic_example
```

Choose a new output directory. This writes row-level OOF, calibration and test tables,
model/preprocessing records and `summary_C.json` for three seeds on each of two datasets.
The script uses the builtin 442-row diabetes dataset with `scaled=False` and a 1,000-row
synthetic Friedman1 dataset. It requires no NASA data or network access.

The main model and OOF-residual risk head are Ridge regressions. Each fold fits its own
scaler, and calibration thresholds use only calibration scores. Score comparisons use
the same point predictions. Results expose realized acceptance, fixed-count error and
discrete AURC-MSE. The retrospective oracle is a diagnostic reference.

The diabetes dataset is used as a tabular software example. The synthetic dataset is
explicitly labeled. Their target units differ. Run outputs must not be pooled across
datasets, and a risk head is not assumed to outperform other scores on every seed.

Source documentation:
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
