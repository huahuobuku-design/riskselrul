"""Standalone metric audit.

Recomputes a fixed set of quantities from the numeric per-sample test tables
only (thresholds, accept decisions, fixed-``k`` masks and error metrics) and
compares them against the stored long metrics table. This module intentionally
imports no experiment module, so it can catch errors in the pipeline it audits.

The basename is taken with :func:`os.path.basename` so the audit runs unchanged
on Windows and POSIX; the original in-project script split on ``"/"`` and failed
on Windows (recorded as a known deviation in the source project).
"""

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd


def verify(out="../results"):
    """Recompute and compare; returns ``{"checked": n, "mismatches": bad}``."""
    M = pd.read_csv(os.path.join(out, "metrics_long.csv"), float_precision="round_trip")
    bad = 0
    n = 0
    for f in sorted(glob.glob(os.path.join(out, "test_*_st*.csv"))):
        ds, s, st = os.path.basename(f)[5:-4].split("_")
        seed = int(s[1:])
        st = int(st[2:])
        T = pd.read_csv(f, float_precision="round_trip")
        preds = [c[5:] for c in T.columns if c.startswith("pred_")]
        for (pm, scen), g in [((pm, sc), T[T.scen == sc])
                              for pm in preds for sc in T.scen.unique()]:
            p = g[f"pred_{pm}"].values
            t = g.y_raw.values
            for h in ["H_log", "H_sq", "pred_only", "fault_indicator"]:
                r = M[(M.ds == ds) & (M.seed == seed) & (M.stream == st) &
                      (M.pred == pm) & (M["head"] == h) & (M.scen == scen)].iloc[0]
                s_ = g[f"score_{pm}_{h}"].values
                sha = np.array([hashlib.sha256(e.encode()).hexdigest() for e in g.engine])
                o = np.lexsort((np.zeros(len(s_)), sha, s_))[:80]
                oo = np.argsort(np.abs(p - t), kind="stable")[:80]
                acc = g[f"acc_{pm}_{h}"].values == 1
                chk = {"full.MSE": np.mean((p - t) ** 2),
                       "k80.sel.MSE": np.mean((p[o] - t[o]) ** 2),
                       "k80.oracle.MSE": np.mean((p[oo] - t[oo]) ** 2),
                       "thr.k": acc.sum(),
                       "full.frac0": np.mean(p == 0)}
                assert np.array_equal(
                    np.sort(np.where(g[f"sel80_{pm}_{h}"].values == 1)[0]), np.sort(o))
                for k, v in chk.items():
                    n += 1
                    if not np.isclose(r[k], v, rtol=1e-9, atol=1e-9):
                        bad += 1
                        print("MISMATCH", f, pm, h, scen, k, r[k], v)
    print(json.dumps(dict(checked=n, mismatches=bad)))
    return dict(checked=n, mismatches=bad)
