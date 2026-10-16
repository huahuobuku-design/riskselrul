"""riskselrul -- risk-aware selective remaining-useful-life prediction.

A reproducible toolkit for selective RUL prediction on the NASA C-MAPSS turbofan
degradation data. It ships an honest evaluation protocol rather than a novel
algorithm: the coverage / error trade-off, fixed-``k`` diagnostics against random
selection, label-aware oracle lower bounds, an engine-cluster bootstrap, and an
independent numeric audit.

Layout:

* :mod:`riskselrul.data`       -- C-MAPSS loading, splits and features
* :mod:`riskselrul.faults`     -- sensor-fault scenarios and deterministic RNG
* :mod:`riskselrul.models`     -- primary predictors and error-risk heads
* :mod:`riskselrul.selective`  -- orderings, decision rules and metrics
* :mod:`riskselrul.evaluation` -- pipeline, bootstrap and audit
* :mod:`riskselrul.deploy`     -- pure-NumPy inference from lean states

The package does not claim algorithmic novelty for the error-risk-head plus
quantile-rejection design; see ``docs/limitations.md``.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
