"""Independent arithmetic audit. No imports from training or inference modules."""
import hashlib
import json
from pathlib import Path
import numpy as np


def audit_run(run_directory):
    import pandas as pd
    root = Path(run_directory)
    metrics = pd.read_csv(root/"metrics_long.csv", float_precision="round_trip")
    tables = sorted(root.glob("test_*_st*.csv"))
    if not tables or metrics.empty:
        raise ValueError("Run has no numeric test tables or metrics")
    checked, failures, direction_rows = 0, [], []
    for path in tables:
        ds, seed, stream = path.stem[5:].split("_")
        seed, stream = int(seed[1:]), int(stream[2:])
        table = pd.read_csv(path, float_precision="round_trip")
        thresholds = json.loads((root/f"thresholds_{ds}_s{seed}.json").read_text())
        calibration = pd.read_csv(root/f"calib_{ds}_s{seed}.csv", float_precision="round_trip")
        predictors = [name[5:] for name in table.columns if name.startswith("pred_")]
        clean = table[table.scen == "clean"].set_index("engine")

        def check(name, actual, expected):
            nonlocal checked
            checked += 1
            if not np.allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True):
                failures.append(dict(table=path.name, check=name))

        for predictor in predictors:
            for head in ("H_log", "H_sq", "pred_only", "fault_indicator"):
                threshold = thresholds[predictor][head]
                check("calibration_quantile", threshold, np.quantile(calibration[f"score_{predictor}_{head}"], .8))
                for scenario, frame in table.groupby("scen", sort=False):
                    if frame.engine.duplicated().any():
                        raise ValueError("Expected one terminal prediction per engine in each scenario")
                    row = metrics[(metrics.ds == ds) & (metrics.seed == seed) & (metrics.stream == stream) &
                                  (metrics.pred == predictor) & (metrics["head"] == head) & (metrics.scen == scenario)]
                    if len(row) != 1:
                        raise ValueError("Missing or duplicate metric cell")
                    row = row.iloc[0]
                    prediction, target = frame[f"pred_{predictor}"].to_numpy(), frame.y_raw.to_numpy()
                    score = frame[f"score_{predictor}_{head}"].to_numpy()
                    error2 = (prediction-target)**2
                    keys = np.array([hashlib.sha256(v.encode()).hexdigest() for v in frame.engine])
                    order = np.lexsort((np.zeros(len(score)), keys, score))
                    oracle = np.argsort(np.abs(prediction-target), kind="stable")
                    check("full_mse", row["full.MSE"], error2.mean())
                    accepted = score <= threshold
                    check("threshold", row["thr.thr"], threshold)
                    check("accepted_count", row["thr.k"], accepted.sum())
                    check("accepted_flags", frame[f"acc_{predictor}_{head}"], accepted)
                    for k in (20, 40, 60, 80, 100):
                        check(f"k{k}_mse", row[f"k{k}.sel.MSE"], error2[order[:k]].mean())
                        check(f"k{k}_oracle", row[f"k{k}.oracle.MSE"], error2[oracle[:k]].mean())
                    mask80 = np.zeros(len(score), dtype=int)
                    mask80[order[:80]] = 1
                    check("fixed80_flags", frame[f"sel80_{predictor}_{head}"], mask80)
                    if not accepted.any():
                        check("zero_accept_risk_is_NA", row["thr.ret_MSE"], np.nan)
                    else:
                        check("retained_mse", row["thr.ret_MSE"], error2[accepted].mean())
                    if scenario != "clean":
                        anchor = clean.loc[frame.engine]
                        delta_error = np.abs(prediction-target)-np.abs(anchor[f"pred_{predictor}"].to_numpy()-anchor.y_raw.to_numpy())
                        delta_score = score-anchor[f"score_{predictor}_{head}"].to_numpy()
                        valid = np.abs(delta_error) > 1e-10
                        success = (np.abs(delta_score) > 1e-10) & (np.sign(delta_score) == np.sign(delta_error))
                        direction_rows.append(dict(dataset=ds, seed=seed, stream=stream, predictor=predictor,
                            head=head, scenario=scenario, denominator=int(valid.sum()),
                            direction_agreement=float(success[valid].mean()) if valid.any() else None,
                            constant_increase_agreement=float((delta_error[valid] > 0).mean()) if valid.any() else None))
    report = dict(checked=checked, mismatches=len(failures), failures=failures,
                  scope="Independent numeric recomputation only; not an execution-history or leakage proof")
    return report, direction_rows
