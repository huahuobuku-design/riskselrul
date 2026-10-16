"""End-to-end training and selective-evaluation pipeline.

Stage 1 fits the four primary predictors and the error-risk heads for every
``(dataset, train seed)``, evaluates the fixed evaluation fault streams, and
writes per-sample tables, per-predictor thresholds and the long metrics table.
Stage 2 runs the pre-specified paired engine-cluster bootstrap on the main
contrast.

Metrics are averaged with equal weight over (seed x stream) combinations, never
over predictions, scores or labels.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .. import config
from ..data import cmapss
from ..faults.rng import engine_order_key, rng_for
from ..faults.scenarios import SENS, TEST_SCEN, TRAIN_SCEN, draw, parse
from ..models.predictors import fit_bundle, observe
from ..models.risk_heads import HEADS, fit_heads, score
from ..selective.decision import cell, order
from ..selective.metrics import anchor, flat, strata_metrics
from . import bootstrap as BS


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bounds(tag, units, heldout, B):
    return dict(split=tag, train_units=units, heldout_units=heldout,
                MU=B["MU"].tolist(), SD=B["SD"].tolist(), zero_sd=B["zero_sd"],
                scaler_mean=B["scl"].mean_.tolist(), scaler_scale=B["scl"].scale_.tolist(),
                n_aug=len(B["aug_meta"]))


def trunc_rows(ds, seed, sp, stream, df, units, B, scen_list, ns, fold=None):
    """Evaluate the given engines at 10 truncation points under each scenario.

    Truncation ends are drawn per engine and shared across scenarios; the raw
    truth is the uncapped ``T - end_cycle``.
    """
    R = []
    for u in units:
        a = df[df.unit == u][SENS].values
        T = len(a)
        eng = f"{ns}:{u}"
        rng, _ = rng_for(ds, seed, sp, "trunc", stream, eng, "na", "na")
        ends = sorted(rng.choice(np.arange(1, T + 1), 10, replace=False).tolist())
        for d, e in enumerate(ends):
            w = cmapss.window(a, e)
            for sc in scen_list:
                on, pos, k = (None, None, None) if sc == "clean" else draw(ds, seed, sp, stream, eng, e, parse(sc)[1])
                X, P, fi = observe(B, w, sc, on, pos)
                R.append(dict(engine=eng, fold=fold, draw=d, wend=e, scen=sc,
                              y_raw=float(T - e), onset=on, pos=pos, key=k, fi=fi, X=X,
                              **{f"raw_{m}": P[m] for m in P}))
    return R


def _table(rows, extra, model_ids):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "X"} for r in rows])
    Xd = pd.DataFrame(np.array([r["X"] for r in rows]), columns=[f"x{i}" for i in range(56)])
    for m in model_ids:
        df[f"pred_{m}"] = np.maximum(df[f"raw_{m}"], 0)
    for k, v in extra.items():
        df[k] = v
    return pd.concat([df, Xd], axis=1)


def run(datasets=None, seeds=None, streams=None, reps=None, out="../results", budget=None):
    """Run stage 1 and, time permitting, stage 2. Returns the run summary dict."""
    datasets = list(datasets or config.DATASETS)
    seeds = list(seeds or config.TRAIN_SEEDS)
    streams = list(streams or config.TEST_STREAMS)
    reps = int(reps or config.BOOTSTRAP_REPS)
    budget = float(budget if budget is not None else 7200)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    json.dump(dict(start_utc=_utc(), data_root=cmapss.ROOT, datasets=datasets,
                   seeds=seeds, streams=streams, reps=reps),
              open(f"{out}/run_info.json", "w"), indent=1)
    BOOTCELLS, MET, ANC = {}, [], []
    status = "complete"
    preds = []
    with threadpool_limits(1):
        for ds in datasets:
            tr, te, rul = cmapss.load(ds)
            for seed in seeds:
                tag = f"{ds}_s{seed}"
                S = cmapss.split(seed)
                json.dump(S, open(f"{out}/split_{tag}.json", "w"))
                oof, bounds = [], []
                for j, fu in enumerate(cmapss.folds(S["train70"])):
                    B = fit_bundle(ds, seed, tr, [u for u in S["train70"] if u not in fu], f"fold{j}")
                    assert not set(fu) & set(B["units"])
                    bounds.append(_bounds(f"fold{j}", B["units"], fu, B))
                    pd.DataFrame(B["aug_meta"], columns=["engine", "wend", "scen", "onset", "pos"]) \
                        .assign(split=f"fold{j}").to_csv(f"{out}/augfaults_{tag}_fold{j}.csv", index=False)
                    oof += trunc_rows(ds, seed, f"fold{j}", config.OOF_STREAM, tr, fu, B,
                                      TRAIN_SCEN, "train", fold=j)
                F = fit_bundle(ds, seed, tr, S["train70"], "final")
                bounds.append(_bounds("final", F["units"], None, F))
                pd.DataFrame(F["aug_meta"], columns=["engine", "wend", "scen", "onset", "pos"]) \
                    .assign(split="final").to_csv(f"{out}/augfaults_{tag}_final.csv", index=False)
                json.dump(bounds, open(f"{out}/fit_boundaries_{tag}.json", "w"))
                cal = trunc_rows(ds, seed, "final", config.CALIB_STREAM, tr, S["calib15"], F,
                                 TRAIN_SCEN, "train")
                tests = {}
                for st in streams:
                    R = []
                    for u in range(1, 101):
                        a2 = te[te.unit == u][SENS].values
                        e = len(a2)
                        w = cmapss.window(a2, e)
                        eng = f"test:{u}"
                        for sc in TEST_SCEN:
                            on, pos, k = (None, None, None) if sc == "clean" else draw(ds, seed, "final", st, eng, e, parse(sc)[1])
                            X, P, fi = observe(F, w, sc, on, pos)
                            R.append(dict(engine=eng, draw=0, wend=e, scen=sc,
                                          y_raw=float(rul[u - 1]), onset=on, pos=pos,
                                          key=k, fi=fi, X=X, **{f"raw_{m}": P[m] for m in P}))
                    tests[st] = R
                oofdf = _table(oof, {}, list(F["M"]))
                oofdf["fold_train_units"] = [json.dumps(bounds[f]["train_units"]) for f in oofdf.fold]
                assert all(int(r.split(":")[1]) not in json.loads(l)
                           for r, l in zip(oofdf.engine, oofdf.fold_train_units))  # no self-engine
                calx, testx, thrs = {}, {st: {} for st in streams}, {}
                for pm in F["M"]:
                    H = fit_heads(oof, seed, pm)
                    sc_cal = score(H, pm, cal)
                    thr = {h: float(np.quantile(sc_cal[h], config.COVERAGE, method="linear")) for h in HEADS}
                    thrs[pm] = thr
                    for h in HEADS:
                        calx[f"score_{pm}_{h}"] = sc_cal[h]
                        calx[f"acc_{pm}_{h}"] = (sc_cal[h] <= thr[h]).astype(int)
                    for st, R in tests.items():
                        SC = score(H, pm, R)
                        eng_all = np.array([r["engine"] for r in R])
                        scen_all = np.array([r["scen"] for r in R])
                        for h in HEADS:
                            testx[st][f"score_{pm}_{h}"] = SC[h]
                            testx[st][f"acc_{pm}_{h}"] = (SC[h] <= thr[h]).astype(int)
                            testx[st][f"sel80_{pm}_{h}"] = np.zeros(len(R), int)
                        ci = np.where(scen_all == "clean")[0]
                        pc = np.maximum([R[i][f"raw_{pm}"] for i in ci], 0)
                        tc = np.array([R[i]["y_raw"] for i in ci])
                        for scen in TEST_SCEN:
                            idx = np.where(scen_all == scen)[0]
                            assert list(eng_all[idx]) == [f"test:{u}" for u in range(1, 101)]
                            p = np.maximum([R[i][f"raw_{pm}"] for i in idx], 0)
                            t = np.array([R[i]["y_raw"] for i in idx])
                            en = list(eng_all[idx])
                            for h in HEADS:
                                s = SC[h][idx]
                                c = cell(p, t, s, thr[h], en, np.zeros(len(idx)))
                                c["strata_full"] = strata_metrics(p, t)
                                o80 = order(s, en, np.zeros(len(idx)))[:config.MAIN_K]
                                testx[st][f"sel80_{pm}_{h}"][idx[o80]] = 1
                                c["strata_sel80"] = strata_metrics(p[o80], t[o80])
                                MET.append(dict(ds=ds, seed=seed, stream=st, pred=pm, head=h, scen=scen, **flat(c)))
                                if scen != "clean":
                                    A_ = anchor(s, SC[h][ci], p - t, pc - tc)
                                    ANC.append(dict(ds=ds, seed=seed, stream=st, pred=pm, head=h, scen=scen,
                                                    **{k: v for k, v in A_.items() if k != "per_engine_agree"}))
                                    key = (ds, h, scen)
                                    BOOTCELLS.setdefault(key, {}).setdefault((seed, st), {})[pm] = dict(
                                        p=p, t=t, s=s, sha=np.array([engine_order_key(x) for x in en]),
                                        eng_idx=np.arange(100))
                                    keyc = (ds, h, "clean")
                                    BOOTCELLS.setdefault(keyc, {}).setdefault((seed, st), {})[pm] = dict(
                                        p=pc, t=tc, s=SC[h][ci],
                                        sha=np.array([engine_order_key(f"test:{u}") for u in range(1, 101)]),
                                        eng_idx=np.arange(100))
                oofdf.to_csv(f"{out}/oof_{tag}.csv", index=False)
                _table(cal, calx, list(F["M"])).to_csv(f"{out}/calib_{tag}.csv", index=False)
                for st in streams:
                    _table(tests[st], testx[st], list(F["M"])).to_csv(f"{out}/test_{tag}_st{st}.csv", index=False)
                json.dump(thrs, open(f"{out}/thresholds_{tag}.json", "w"), indent=1)
                pd.DataFrame(MET).to_csv(f"{out}/metrics_long.csv", index=False)
                pd.DataFrame(ANC).to_csv(f"{out}/anchor.csv", index=False)
                preds = list(F["M"])
                print(tag, "stage1 done", round(time.time() - t0), "s", flush=True)
        M = pd.DataFrame(MET)
        num = [c for c in M.columns if c not in ("ds", "seed", "stream", "pred", "head", "scen")
               and pd.api.types.is_numeric_dtype(M[c])]
        M.groupby(["ds", "pred", "head", "scen"])[num].mean().reset_index() \
            .to_csv(f"{out}/metrics_avg_combos.csv", index=False)
        rows = []
        for (ds, h, scen), cells in sorted(BOOTCELLS.items()):
            if scen == "clean" or h not in ("H_log", "H_sq"):
                continue
            if time.time() - t0 > budget:
                status = "incomplete_timeout"
                break
            breps = BS.run(cells, BOOTCELLS[(ds, h, "clean")], preds,
                           reps=reps, seed=config.BOOTSTRAP_SEED)
            arr = {pm: np.array([r[pm] for r in breps]) for pm in preds}
            for pm in preds:
                for q, nm in enumerate(["selection_regret", "base_floor", "total"]):
                    rows.append(dict(ds=ds, head=h, scen=scen, contrast=pm, quantity=nm,
                                     lo=np.percentile(arr[pm][:, q], 2.5),
                                     hi=np.percentile(arr[pm][:, q], 97.5)))
            for i, p1 in enumerate(preds):
                for p2 in preds[i + 1:]:
                    for q, nm in enumerate(["selection_regret", "base_floor", "total"]):
                        d = arr[p1][:, q] - arr[p2][:, q]
                        rows.append(dict(ds=ds, head=h, scen=scen, contrast=f"{p1}-{p2}",
                                         quantity=nm, lo=np.percentile(d, 2.5),
                                         hi=np.percentile(d, 97.5)))
        pd.DataFrame(rows).assign(status=status, reps=reps).to_csv(f"{out}/bootstrap_intervals.csv", index=False)
    summary = dict(end_utc=_utc(), elapsed_s=round(time.time() - t0, 1), bootstrap_status=status)
    json.dump(summary, open(f"{out}/run_end.json", "w"), indent=1)
    print("done", status, round(time.time() - t0), "s", flush=True)
    return summary
