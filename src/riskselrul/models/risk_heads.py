"""Label-free error-risk heads.

Every head consumes out-of-fold (OOF) residuals of a fixed primary predictor.
No test label is ever used for fitting a head or for selecting a threshold.

=================  ============================================================
``H_log``          fits ``log1p(|y - pred|)``
``H_sq``           fits ``(y - pred)^2 / CAP^2``
``pred_only``      same squared target, prediction as the only input
``fault_indicator``  no fit; cheap NaN/constant-run statistic of the window
=================  ============================================================
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .. import config

HEADS = ["H_log", "H_sq", "pred_only", "fault_indicator"]


def _gbdt(seed):
    return GradientBoostingRegressor(loss="squared_error", n_estimators=100, max_depth=2,
                                     learning_rate=.05, min_samples_leaf=20, subsample=1.0,
                                     random_state=seed)


def fit_heads(oof, seed, pm):
    """Fit the three fitted heads for primary predictor ``pm`` from OOF rows."""
    X = np.array([r["X"] for r in oof])
    p = np.maximum(np.array([r[f"raw_{pm}"] for r in oof]), 0)
    y = np.array([r["y_raw"] for r in oof])
    Z = np.c_[X, p]
    return {"H_log": _gbdt(seed).fit(Z, np.log1p(np.abs(y - p))),
            "H_sq": _gbdt(seed).fit(Z, (y - p) ** 2 / config.CAP ** 2),
            "pred_only": _gbdt(seed).fit(p[:, None], (y - p) ** 2 / config.CAP ** 2)}


def score(H, pm, rows):
    """Score rows with all four heads; the fault indicator needs no fitted model."""
    X = np.array([r["X"] for r in rows])
    p = np.maximum(np.array([r[f"raw_{pm}"] for r in rows]), 0)
    Z = np.c_[X, p]
    return {"H_log": H["H_log"].predict(Z),
            "H_sq": H["H_sq"].predict(Z),
            "pred_only": H["pred_only"].predict(p[:, None]),
            "fault_indicator": np.array([r["fi"] for r in rows])}
