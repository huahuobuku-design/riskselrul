"""Pure-NumPy inference from pre-built lean deployment states.

A lean state is a small ``.npz`` holding numeric arrays only: no pickled objects,
no covariance matrix and no bootstrap ensemble. It maps a raw window of shape
``(n, 30, 21)`` -- values in original engineering units, NaN allowed -- to a
predicted RUL, an error-risk score, and an accept / reject decision.

Head widths:

=====  ==================================================================
57     linear risk head on the 56 features plus the prediction
99     adds NaN fraction, constant-run score and a volatility ratio
101    adds Mahalanobis distance and bootstrap spread (full reference)
=====  ==================================================================

The released states cover the 57- and 99-dimensional linear heads for FD001 and
FD003. Labels, engine identifiers, scenario names and fault metadata are never
inputs to this module.

State keys (see the released ``states_index.csv``)::

    indices, raw_MU21, raw_MU, raw_SD,
    scaler_mean, scaler_scale, ridge_coef, ridge_intercept,
    risk_mean, risk_scale, risk_coef, risk_intercept,
    threshold_condition, threshold_coverage, threshold_value
"""

import numpy as np

T_AX = np.arange(30) - 14.5


def features(raw, state, width=57):
    """Return ``(prediction, risk_head_inputs)`` for a batch of raw windows.

    ``risk_head_inputs`` has ``width`` columns: 57 = features + prediction,
    99 = the above plus three fault indicators, 101 = the above plus two
    uncertainty terms.
    """
    raw = np.asarray(raw, dtype=float)
    assert raw.ndim == 3 and raw.shape[1:] == (30, 21)

    imp = raw.copy()
    ii = np.where(np.isnan(imp))
    imp[ii] = state["raw_MU21"][ii[2]]
    ix = state["indices"]
    w = imp[:, :, ix]
    z = (w - state["raw_MU"]) / state["raw_SD"]

    # Window-wise arithmetic, identical across the timing variants.
    f = np.array([np.concatenate([a.mean(0), a.std(0), a[-1],
                                  T_AX @ (a - a.mean(0)) / (T_AX @ T_AX)]) for a in z])
    X = (f - state["scaler_mean"]) / state["scaler_scale"]
    p = np.maximum(X @ state["ridge_coef"] + state["ridge_intercept"][0], 0)

    if width == 57:
        return p, np.column_stack([X, p])

    rn = raw[:, :, ix]
    nanf = np.isnan(rn).mean(1)
    eq = rn[:, 1:, :] == rn[:, :-1, :]
    run = np.zeros((len(raw), len(ix)))
    cur = run.copy()
    for t in range(29):
        cur = np.where(eq[:, t, :], cur + 1, 0)
        run = np.maximum(run, cur)
    runf = np.where(run + 1 >= 5, (run + 1) / 30, 0)
    ratio = w[:, -10:, :].std(1) / (w[:, :20, :].std(1) + 1e-9)

    if width == 99:
        return p, np.column_stack([X, nanf, runf, ratio, p])

    d = X - state["maha_location"]
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, state["maha_precision"], d))
    spread = (X @ state["boot_coef"].T + state["boot_intercept"]).std(1)
    return p, np.column_stack([X, nanf, runf, ratio, maha, spread, p])


def risk(X, state):
    """Error-risk score from the risk-head inputs."""
    return ((X - state["risk_mean"]) / state["risk_scale"]) @ state["risk_coef"] \
        + state["risk_intercept"][0]


def load(path):
    """Load a lean state, asserting it holds numeric arrays only."""
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    assert all(a.dtype.kind != "O" for a in s.values())
    assert not any("maha" in k or "boot" in k for k in s)
    return s


def predict(raw, state):
    """Return ``(predicted_rul, risk_score)`` for a batch of raw windows."""
    p, X = features(raw, state, len(state["risk_coef"]))
    return p, risk(X, state)


def decide(raw, state, condition="C2", coverage=0.8):
    """Return ``(predicted_rul, risk_score, accept)`` using a frozen threshold.

    ``condition`` and ``coverage`` must match a frozen calibration entry stored
    in the state; the threshold is never chosen from fault truth at run time.
    """
    p, q = predict(raw, state)
    ix = np.where((state["threshold_condition"] == condition)
                  & (state["threshold_coverage"] == coverage))[0]
    if len(ix) != 1:
        raise ValueError("Requested frozen calibration condition is unavailable for this head")
    return p, q, q <= state["threshold_value"][ix[0]]
