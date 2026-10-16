"""NumPy-only diagnostics; selection never sees target labels.

Oracle values are retrospective diagnostics. The expected MSE of uniform
random retention equals full-sample MSE; this identity does not apply to RMSE.
"""
import hashlib
import numpy as np


def _vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


def _pair(prediction, target):
    prediction, target = _vector(prediction, "prediction"), _vector(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target lengths differ")
    return prediction, target


def prediction_metrics(prediction, target, overestimate_margin=20.0):
    prediction, target = _pair(prediction, target)
    if overestimate_margin < 0 or not np.isfinite(overestimate_margin):
        raise ValueError("overestimate_margin must be finite and nonnegative")
    if len(target) == 0:
        return dict(n=0, mse=None, rmse=None, mae=None, mean_error=None, overestimate_rate=None)
    error = prediction-target
    return dict(n=len(target), mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), mean_error=float(np.mean(error)),
                overestimate_rate=float(np.mean(error > overestimate_margin)))


def calibrate_threshold(calibration_scores, acceptance_rate=0.8):
    scores = _vector(calibration_scores, "calibration_scores")
    if not len(scores) or not 0 < acceptance_rate <= 1:
        raise ValueError("nonempty calibration scores and acceptance_rate in (0,1] are required")
    return float(np.quantile(scores, acceptance_rate, method="linear"))


def stable_risk_order(scores, sample_ids=None):
    scores = _vector(scores, "scores")
    sample_ids = [str(i) for i in range(len(scores))] if sample_ids is None else list(sample_ids)
    if len(sample_ids) != len(scores):
        raise ValueError("sample_ids and scores lengths differ")
    keys = np.array([hashlib.sha256(str(v).encode("utf-8")).hexdigest() for v in sample_ids])
    return np.lexsort((np.arange(len(scores)), keys, scores))


def fixed_k_metrics(prediction, target, scores, k, sample_ids=None):
    prediction, target = _pair(prediction, target)
    scores = _vector(scores, "scores")
    n = len(target)
    if len(scores) != n or isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and n; scores must match n")
    chosen = stable_risk_order(scores, sample_ids)[:k]
    # Labels enter only the retrospective oracle calculation below.
    oracle = np.argsort(np.abs(prediction-target), kind="stable")[:k]
    full = prediction_metrics(prediction, target)
    selected = prediction_metrics(prediction[chosen], target[chosen])
    ideal = prediction_metrics(prediction[oracle], target[oracle])
    return dict(k=int(k), n=n, acceptance_rate=float(k/n) if n else 0.0,
                retained=selected, oracle=ideal, full=full,
                random_expected_mse=full["mse"] if k else None,
                selection_regret=None if not k else float(max(0.0, selected["mse"]-ideal["mse"])),
                selected_indices=chosen.tolist())


def risk_coverage_curve(prediction, target, scores, sample_ids=None):
    prediction, target = _pair(prediction, target)
    scores = _vector(scores, "scores")
    if len(scores) != len(target):
        raise ValueError("scores and target lengths differ")
    n = len(target)
    if not n:
        return []
    order = stable_risk_order(scores, sample_ids)
    e2 = (prediction-target)**2
    counts = np.arange(1, n+1)
    retained = np.cumsum(e2[order])/counts
    oracle = np.cumsum(np.sort(e2, kind="stable"))/counts
    return [dict(k=int(k), acceptance_rate=float(k/n), retained_mse=float(m),
                 retained_rmse=float(np.sqrt(m)), oracle_mse=float(o), random_expected_mse=float(e2.mean()))
            for k, m, o in zip(counts, retained, oracle)]
