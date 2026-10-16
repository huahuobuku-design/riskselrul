"""C-MAPSS loading, engine splits, folds and windowing.

The released experiments use the FD001 and FD003 subsets of the NASA C-MAPSS
turbofan degradation data set, but this loader is dataset-agnostic.

Expected layout under the data root (``data/raw`` by default)::

    train_FD001.txt  test_FD001.txt  RUL_FD001.txt
    train_FD003.txt  test_FD003.txt  RUL_FD003.txt

Text files are whitespace separated with 26 columns:
``unit, cycle, op1, op2, op3, s1..s21``.
"""

import hashlib
import os

import numpy as np
import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

_DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw"))
ROOT = os.environ.get("RISKSELRUL_DATA_ROOT", _DEFAULT_ROOT)


def set_root(p):
    """Override the directory holding ``train_{ds}.txt`` / ``test_{ds}.txt`` / ``RUL_{ds}.txt``."""
    global ROOT
    ROOT = p


def sha(p):
    """SHA-256 of a file, used for input manifests and freeze records."""
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(ds):
    """Load one C-MAPSS subset and return ``(train_df, test_df, rul_array)``."""
    tr = pd.read_csv(f"{ROOT}/train_{ds}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(f"{ROOT}/test_{ds}.txt", sep=r"\s+", header=None, names=COLS)
    rul = np.loadtxt(f"{ROOT}/RUL_{ds}.txt")
    return tr, te, rul


def split(seed):
    """70/15/15 engine split (train / validation / calibration) from a single seed."""
    p = np.random.default_rng(seed).permutation(np.arange(1, 101))
    return dict(train70=p[:70].tolist(), val15=p[70:85].tolist(), calib15=p[85:].tolist())


def folds(train70):
    """Five folds of 14 engines each, in permutation order."""
    return [train70[14 * j:14 * j + 14] for j in range(5)]


def window(a, e, W=30):
    """Trailing window ending at cycle index ``e``; short windows are front-padded by repeating row 0."""
    w = a[max(0, e - W):e]
    return w if len(w) == W else np.vstack([np.repeat(w[:1], W - len(w), 0), w])
