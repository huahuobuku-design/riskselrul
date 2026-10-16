# Architecture and protocol boundaries

`inspect-data / extract-data -> reproduce -> numeric tables -> audit / curve`

The independent `predict` path is `raw (n,30,21) -> archived numeric WP04A state -> RUL, risk, threshold decision`. It requires only NumPy. It does not call the training module or load object models.

The training path runs a migrated WP05A numerical profile in a subprocess. The library does not alter the caller's thread settings. Training is limited to one CPU thread through environment variables and threadpoolctl. Four point predictors share raw windows and fault streams. Their training targets are capped at 125; risk targets and reported test error use uncapped RUL. Lower risk scores are preferred.

For each seed, a deterministic permutation splits the 100 original training engines into 70 fitting, 15 unused validation, and 15 calibration engines. Five OOF folds each hold out 14 of the fitting engines. Sensor normalization and feature scaling are fit only on the relevant clean fitting engines. Risk heads are trained from held-out-engine residuals. Test engines do not influence fitting or calibration.

Training augmentation uses sensor group A; B is held out from augmentation. The 21 synthetic scenarios cover clean, dropout, stuck, signed bias, and signed drift. Identical hash-keyed random streams couple matched comparisons. These scenarios do not represent all real sensor failures.

The `audit` module reads numbers and recalculates calibration thresholds, acceptance flags, errors, fixed-k decisions, and oracle errors without importing the training or inference implementation. Arithmetic agreement cannot establish that an upstream execution history or a scientific causal claim is correct.
