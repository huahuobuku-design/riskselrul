"""Feature construction for the primary predictors and the risk heads.

Each ``(30, 14)`` raw window becomes a 56-dimensional vector: per-channel mean,
per-channel standard deviation, last cycle value, and least-squares slope over a
centred time axis. Standardization uses the per-channel mean / ddof-1 standard
deviation estimated on the clean training engines only; channels with an invalid
(practically zero or non-finite) standard deviation fall back to unit scaling.
"""

import numpy as np

from ..faults.scenarios import SENS

T_AX = np.arange(30) - 14.5


def sensor_stats(tr, units):
    """Per-channel mean and ddof-1 standard deviation over the given training engines."""
    v = tr[tr.unit.isin(units)][SENS]
    return v.mean().values, v.std(ddof=1).values


def feats(w, MU, SD):
    """Map one ``(30, 14)`` window to its 56-dimensional feature vector."""
    sd = np.where(np.isfinite(SD) & (SD > 1e-12), SD, 1.0)
    z = (w - MU) / sd
    return np.concatenate([z.mean(0), z.std(0), z[-1],
                           (T_AX @ (z - z.mean(0))) / (T_AX @ T_AX)])
