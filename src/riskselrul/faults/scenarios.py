"""Sensor fault generator.

A raw window is an array of shape ``(30, 14)`` in config sensor order (the 14
informative C-MAPSS sensors). Generation is pure NumPy and fully deterministic
given the RNG key produced by :mod:`riskselrul.faults.rng`.

Two channel groups are used throughout the project:

* group ``A`` -- the seven channels seen by the training-time fault generator;
* group ``B`` -- seven held-out channels never injected during training.

Supported fault types are ``dropout``, ``stuck``, ``bias`` and ``drift``.
"""

import numpy as np

from .rng import rng_for

SENS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
A = ["s2", "s4", "s8", "s11", "s13", "s15", "s20"]
B = ["s3", "s7", "s9", "s12", "s14", "s17", "s21"]

NCH = {"dropout1": 1, "dropout3": 3, "stuck1": 1, "stuck3": 3, "bias2": 3, "bias4": 3, "drift4": 3}

TRAIN_SCEN = ["clean", "A_dropout1", "A_dropout3", "A_stuck1", "A_stuck3",
              "A_bias+2", "A_bias+4", "A_drift+4"]

TEST_SCEN = (["clean"]
             + [f"{g}_{t}" for g in "AB" for t in ["dropout1", "dropout3", "stuck1", "stuck3"]]
             + [f"{g}_{t}{s}{m}"
                for g in "AB"
                for t, m in [("bias", 2), ("bias", 4), ("drift", 4)]
                for s in "+-"])


def parse(scen):
    """Parse a scenario label, e.g. ``'B_bias-4' -> (group, 'bias4', type, sign, mag)``."""
    if scen == "clean":
        return None
    g, r = scen.split("_")
    for t in ("dropout", "stuck"):
        if r.startswith(t):
            return g, r, t, 1, 0
    t = "bias" if r.startswith("bias") else "drift"
    sign = 1 if r[len(t)] == "+" else -1
    m = int(r[len(t) + 1:])
    return g, f"{t}{m}", t, sign, m


def draw(ds, seed, split, stream, engine, wend, base):
    """Draw onset and within-group positions.

    The key deliberately excludes sign and group, so that ``+``/``-`` and A/B
    variants of the same base fault share onset and channel positions.
    """
    rng, k = rng_for(ds, seed, split, "fault", stream, engine, wend, base)
    on = int(rng.integers(5, 26))
    pos = rng.choice(7, size=NCH[base], replace=False)
    return on, [int(p) for p in pos], k


def apply(raw, scen, on, pos, SD):
    """Inject a fault into a raw ``(30, 14)`` window.

    ``SD`` is the ``(14,)`` raw ddof-1 standard deviation used as the amplitude
    unit (amplitude is 0 when the entry is invalid). Dropout is represented by
    NaN; imputation is a separate step.
    """
    x = np.array(raw, float, copy=True)
    p = parse(scen)
    if p is None:
        return x
    g, base, t, sign, m = p
    grp = A if g == "A" else B
    for q in pos:
        j = SENS.index(grp[q])
        sd = SD[j] if (np.isfinite(SD[j]) and SD[j] > 1e-12) else 0.0
        if t == "dropout":
            x[on:, j] = np.nan
        elif t == "stuck":
            x[on:, j] = x[on, j]
        elif t == "bias":
            x[on:, j] += sign * m * sd
        elif t == "drift":
            x[on:, j] += sign * m * sd * np.linspace(0, 1, 30 - on)
    return x


def impute(x, MU):
    """Replace NaN entries by the per-channel training mean."""
    y = x.copy()
    r, c = np.where(np.isnan(y))
    y[r, c] = np.asarray(MU)[c]
    return y


def fault_indicator(x):
    """Cheap label-free fault indicator on the pre-imputation ``(30, 14)`` window.

    Mean NaN fraction plus mean longest-constant-run score (``L/30`` when the
    longest run ``L >= 5``, else 0).
    """
    nanf = np.isnan(x).mean(0)
    runf = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        L = cur = 1
        for t in range(1, x.shape[0]):
            cur = cur + 1 if x[t, j] == x[t - 1, j] else 1
            L = max(L, cur)
        runf[j] = L / 30 if L >= 5 else 0.0
    return float(nanf.mean() + runf.mean())


def aug_choice(ds, seed, split, engine, wend, stream=2026101601):
    """Pick the augmentation scenario for one clean training window.

    Uniform over the seven non-clean training scenarios; the same choice is
    shared by every predictor regime that uses augmentation.
    """
    rng, k = rng_for(ds, seed, split, "aug_choice", stream, engine, wend, "na")
    return TRAIN_SCEN[1 + int(rng.integers(0, 7))], k
