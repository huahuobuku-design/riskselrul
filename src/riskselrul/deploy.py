"""Frozen WP04A raw-window inference; NumPy only, without object deserialization."""
from importlib.resources import files
from pathlib import Path
import numpy as np

SENSOR_TIME = np.arange(30)-14.5


def available_states():
    return sorted(p.name for p in files("riskselrul").joinpath("assets/states").iterdir() if p.name.endswith(".npz"))


def load_state(path_or_name):
    path = Path(path_or_name)
    if not path.is_file():
        if path.name != str(path_or_name) or path.name not in available_states():
            raise FileNotFoundError(f"Unknown state: {path_or_name}")
        path = files("riskselrul").joinpath("assets/states", path.name)
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    required = {"indices", "raw_MU21", "raw_MU", "raw_SD", "scaler_mean", "scaler_scale", "ridge_coef",
                "ridge_intercept", "risk_mean", "risk_scale", "risk_coef", "risk_intercept",
                "threshold_condition", "threshold_coverage", "threshold_value"}
    missing = required-state.keys()
    if missing or any(a.dtype.kind == "O" for a in state.values()):
        raise ValueError(f"Invalid numeric state; missing keys: {sorted(missing)}")
    width = state["risk_coef"].size
    if width not in (57, 99) or len(state["indices"]) != 14:
        raise ValueError("Only the frozen WP04A 57/99 feature profiles are supported")
    for key in required-{"threshold_condition"}:
        if not np.isfinite(state[key]).all():
            raise ValueError(f"Nonfinite state array: {key}")
    for key in ["raw_SD", "scaler_scale", "risk_scale"]:
        if np.any(state[key] <= 0):
            raise ValueError(f"Nonpositive scale in state: {key}")
    return state


def _features(raw, state):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 3 or raw.shape[1:] != (30, 21) or np.isinf(raw).any():
        raise ValueError("Expected raw windows of shape (n,30,21), finite or NaN")
    if len(raw) == 0:
        return np.empty(0), np.empty((0, state["risk_coef"].size))
    imputed = raw.copy()
    rows, times, channels = np.where(np.isnan(imputed))
    imputed[rows, times, channels] = state["raw_MU21"][channels]
    indices = state["indices"]
    windows = imputed[:, :, indices]
    z = (windows-state["raw_MU"])/state["raw_SD"]
    # Preserve the archived per-window arithmetic, including standard deviation ddof=0.
    features = np.array([np.concatenate([a.mean(0), a.std(0), a[-1],
                         SENSOR_TIME@(a-a.mean(0))/(SENSOR_TIME@SENSOR_TIME)]) for a in z])
    x = (features-state["scaler_mean"])/state["scaler_scale"]
    prediction = np.maximum(x@state["ridge_coef"]+state["ridge_intercept"][0], 0)
    if state["risk_coef"].size == 57:
        return prediction, np.column_stack([x, prediction])
    raw_selected = raw[:, :, indices]
    missing_fraction = np.isnan(raw_selected).mean(1)
    equals = raw_selected[:, 1:, :] == raw_selected[:, :-1, :]
    longest = np.zeros((len(raw), len(indices)))
    current = longest.copy()
    for time in range(29):
        current = np.where(equals[:, time, :], current+1, 0)
        longest = np.maximum(longest, current)
    run_fraction = np.where(longest+1 >= 5, (longest+1)/30, 0)
    ratio = windows[:, -10:, :].std(1)/(windows[:, :20, :].std(1)+1e-9)
    return prediction, np.column_stack([x, missing_fraction, run_fraction, ratio, prediction])


def predict(raw_windows, state):
    prediction, x = _features(raw_windows, state)
    risk = ((x-state["risk_mean"])/state["risk_scale"])@state["risk_coef"]+state["risk_intercept"][0]
    return prediction, risk


def decide(raw_windows, state, condition="C2", acceptance_rate=0.8):
    prediction, risk = predict(raw_windows, state)
    matches = np.flatnonzero((state["threshold_condition"] == condition) &
                             (state["threshold_coverage"] == acceptance_rate))
    if len(matches) != 1:
        raise ValueError("This state has no frozen threshold for the requested condition/rate")
    return prediction, risk, risk <= state["threshold_value"][matches[0]]
