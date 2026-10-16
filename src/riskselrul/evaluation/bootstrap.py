"""Paired engine-cluster bootstrap.

Each replicate resamples the 100 test engines with replacement and recomputes,
for every ``(seed, stream)`` combination and predictor, the fixed-``k``
risk-selected MSE, the label-oracle MSE and the full MSE; combinations are then
averaged with equal weight, and finally pairwise predictor differences are
formed. Duplicates are kept; ties are broken by (score, engine hash, draw
position).
"""

import numpy as np


def k_readout(p, t, s, sha, k=80):
    """Fixed-``k`` risk-selected, oracle and full MSE for one cell."""
    e2 = (p - t) ** 2
    dpos = np.arange(len(p))
    o = np.lexsort((dpos, sha, s))[:k]
    oo = np.lexsort((dpos, np.abs(p - t)))[:k]
    return dict(sel=e2[o].mean(), oracle=e2[oo].mean(), full=e2.mean())


def resample(cell, idx):
    """Resample one test cell by engine indices (one terminal row per engine)."""
    assert np.array_equal(np.asarray(cell["eng_idx"]), np.arange(100))
    pos = np.asarray(idx)
    return {k: np.asarray(v)[pos] for k, v in cell.items()}


def run(cells_fault, cells_clean, preds, reps=2000, seed=20261016, k=80):
    """Run the bootstrap.

    ``cells_fault`` / ``cells_clean`` map combo key -> predictor -> cell dict with
    keys ``p, t, s, sha, eng_idx``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, 100, 100)
        row = {}
        for pm in preds:
            acc = []
            for c in cells_fault:
                f = k_readout(**{x: resample(cells_fault[c][pm], idx)[x]
                                 for x in ("p", "t", "s", "sha")}, k=k)
                cl = resample(cells_clean[c][pm], idx)
                fc = np.mean((cl["p"] - cl["t"]) ** 2)
                acc.append((f["sel"] - f["oracle"], f["oracle"] - fc, f["sel"] - fc))
            row[pm] = np.mean(acc, 0)
        out.append(row)
    return out
