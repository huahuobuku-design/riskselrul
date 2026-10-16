"""Primary RUL predictors and the training-augmentation bundle.

Four predictors are produced for every ``(dataset, seed, engine subset)``:

======================  ===============================================
``ridge_clean``         Ridge(alpha=100) on clean windows
``ridge_aug``           Ridge on clean windows plus one fault copy each
``hgb_clean``           HistGradientBoostingRegressor on clean windows
``hgb_aug``             HGB on clean windows plus one fault copy each
======================  ===============================================

The augmented copies are shared between the Ridge and HGB regimes, and the
:class:`~sklearn.preprocessing.StandardScaler` is fitted on clean windows only.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .. import config
from ..data.features import feats, sensor_stats
from ..faults.scenarios import SENS, apply, aug_choice, draw, fault_indicator, impute, parse

PREDICTOR_IDS = ["ridge_clean", "ridge_aug", "hgb_clean", "hgb_aug"]


def fit_bundle(ds, seed, tr, units, sp):
    """Fit the four primary predictors on the given training engines.

    ``units`` is the engine list used for fitting; ``sp`` is a split tag kept for
    RNG addressing. Windows are all full trailing windows ending at cycle
    ``e = 30..T`` of each engine.
    """
    MU, SD = sensor_stats(tr, units)
    Xc, yc, Xa, ya, meta = [], [], [], [], []
    for u in units:
        a = tr[tr.unit == u][SENS].values
        T = len(a)
        eng = f"train:{u}"
        for e in range(30, T + 1):
            w = a[e - 30:e]
            Xc.append(feats(w, MU, SD))
            yc.append(min(T - e, config.CAP))
            sc, _ = aug_choice(ds, seed, sp, eng, e, config.AUG_STREAM)
            on, pos, _k = draw(ds, seed, sp, config.AUG_STREAM, eng, e, parse(sc)[1])
            Xa.append(feats(impute(apply(w, sc, on, pos, SD), MU), MU, SD))
            ya.append(min(T - e, config.CAP))
            meta.append((eng, e, sc, on, pos))
    Xc, yc, Xa, ya = map(np.asarray, (Xc, yc, Xa, ya))
    scl = StandardScaler().fit(Xc)  # clean windows only
    M = {}
    for reg, (X, y) in {"clean": (Xc, yc), "aug": (np.vstack([Xc, Xa]), np.r_[yc, ya])}.items():
        Xs = scl.transform(X)
        M[f"ridge_{reg}"] = Ridge(alpha=100.0).fit(Xs, y)
        M[f"hgb_{reg}"] = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=.05, max_iter=150, max_leaf_nodes=15,
            max_depth=None, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=seed).fit(Xs, y)
    zero = [SENS[j] for j in range(14) if not (np.isfinite(SD[j]) and SD[j] > 1e-12)]
    return dict(MU=MU, SD=SD, scl=scl, M=M, units=list(units), aug_meta=meta, zero_sd=zero)


def observe(B, w, sc, on, pos):
    """Apply an evaluation-time fault to a window.

    Returns ``(X56, {predictor_id: prediction}, fault_indicator)`` where the
    fault indicator is computed on the pre-imputation window.
    """
    x = apply(w, sc, on, pos, B["SD"])
    fi = fault_indicator(x)
    X = B["scl"].transform(feats(impute(x, B["MU"]), B["MU"], B["SD"])[None])[0]
    P = {m: float(B["M"][m].predict(X[None])[0]) for m in B["M"]}
    return X, P, fi
