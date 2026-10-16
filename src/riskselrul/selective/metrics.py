"""Error and selective-prediction metrics.

All metrics operate on per-sample ``(prediction, truth)`` pairs. The selective
quantities compare a risk-ordered selection against (i) an equally sized random
selection and (ii) a label-aware oracle over the same fixed predictions -- the
latter is a diagnostic lower bound, not a deployable method.
"""

import numpy as np
from scipy.stats import spearmanr

from .. import config


def metrics(p, t):
    """Standard error metrics for one ``(prediction, truth)`` subset."""
    e = p - t
    n = len(t)
    if n == 0:
        return dict(n=0, RMSE=None, MAE=None, MSE=None, ME=None,
                    frac0=None, over20=None, under20=None)
    return dict(n=n, RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(abs(e))),
                MSE=float(np.mean(e ** 2)), ME=float(e.mean()),
                frac0=float(np.mean(p == 0)),
                over20=float(np.mean(e > 20)), under20=float(np.mean(e < -20)))


def strata_metrics(p, t):
    """Split metrics by raw-truth strata (``<= CAP`` vs ``> CAP``); offline diagnostic."""
    return {"y_le125": metrics(p[t <= config.CAP], t[t <= config.CAP]),
            "y_gt125": metrics(p[t > config.CAP], t[t > config.CAP])}


def anchor(sf, sc, ef, ec):
    """Same-engine clean anchor.

    Compares the risk change and the absolute-error change of a faulted window
    against its clean counterpart: direction-agreement rate (rows where both
    deltas are non-zero) and Spearman correlation between the two deltas.
    """
    dr = sf - sc
    de = np.abs(ef) - np.abs(ec)
    nz = (dr != 0) & (de != 0)
    agree = float(np.mean(np.sign(dr[nz]) == np.sign(de[nz]))) if nz.any() else None
    rho = None if (np.ptp(dr) == 0 or np.ptp(de) == 0) else float(spearmanr(dr, de).statistic)
    return dict(dir_agree=agree, n_nonzero=int(nz.sum()), spearman=rho,
                const_all_increase=float(np.mean(de > 0)),
                per_engine_agree=np.where(nz, (np.sign(dr) == np.sign(de)).astype(float), np.nan))


def flat(d, pre=""):
    """Flatten a nested dict of scalars into dotted keys; arrays are dropped."""
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, f"{pre}{k}."))
        elif not isinstance(v, np.ndarray):
            o[f"{pre}{k}"] = v
    return o
