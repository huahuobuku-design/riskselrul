"""Selective decision rules: risk ordering, fixed-k selection and thresholds.

Selective prediction accepts a prediction when its error-risk score is below a
threshold fitted on an independent calibration stream, and rejects it otherwise.
Fixed-``k`` selection is kept for diagnosis only: it is a ranking read-out with a
preset retained count, not a deployable online rule.
"""

import numpy as np

from .. import config
from ..faults.rng import engine_order_key
from .metrics import metrics


def order(s, engines, draws):
    """Ascending-risk order with label-free tie-breaks.

    Ties are broken by the engine hash and then the draw position, so no label
    information enters the ordering.
    """
    return np.lexsort((np.asarray(draws),
                       np.array([engine_order_key(e) for e in engines]),
                       np.asarray(s)))


def cell(p, t, s, thr, engines, draws):
    """All metrics for one evaluation cell.

    Returns full-sample metrics, the metrics of the threshold-accepted subset,
    and, for every fixed ``k`` in :data:`riskselrul.config.FIXED_K`, both the
    risk-selected subset and the label-oracle subset over the same predictions.
    """
    out = dict(full=metrics(p, t))
    acc = s <= thr
    out["thr"] = dict(thr=float(thr), k=int(acc.sum()), ratio=float(acc.mean()),
                      **{f"ret_{a}": b for a, b in metrics(p[acc], t[acc]).items()})
    o = order(s, engines, draws)
    oo = np.argsort(np.abs(p - t), kind="stable")
    for k in config.FIXED_K:
        out[f"k{k}"] = dict(sel=metrics(p[o[:k]], t[o[:k]]),
                            oracle=metrics(p[oo[:k]], t[oo[:k]]))
    return out
