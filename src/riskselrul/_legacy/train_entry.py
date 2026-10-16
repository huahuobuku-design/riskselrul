"""WP05A phase-2 entry (FROZEN; not executed in phase 1). Usage: python train_entry.py --run-phase2 --out ../results
Each (dataset, train seed, test stream) is processed independently: prediction, ranking, threshold, metrics. Metrics are averaged
with equal weight only after that (never predictions/scores/labels)."""
import os
for v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]: os.environ[v]="1"
import sys, json, time, argparse, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from .rng_keys import rng_for, engine_order_key
from .faults import SENS, TRAIN_SCEN, TEST_SCEN, parse, draw, apply, impute, fault_indicator, aug_choice
from . import data as DATA
from .data import load, split, folds, window, sensor_stats, feats
C=json.load(open(os.path.join(os.path.dirname(__file__),"config.json")))
CAP=125; ST=C["streams"]
def fit_bundle(ds,seed,tr,units,sp):
    MU,SD=sensor_stats(tr,units); Xc,yc,Xa,ya,meta=[],[],[],[],[]
    for u in units:
        a=tr[tr.unit==u][SENS].values; T=len(a); eng=f"train:{u}"
        for e in range(30,T+1):
            w=a[e-30:e]; Xc.append(feats(w,MU,SD)); yc.append(min(T-e,CAP))
            sc,_=aug_choice(ds,seed,sp,eng,e,ST["aug"]); on,pos,k=draw(ds,seed,sp,ST["aug"],eng,e,parse(sc)[1])
            Xa.append(feats(impute(apply(w,sc,on,pos,SD),MU),MU,SD)); ya.append(min(T-e,CAP)); meta.append((eng,e,sc,on,pos))
    Xc,yc,Xa,ya=map(np.asarray,(Xc,yc,Xa,ya)); scl=StandardScaler().fit(Xc)  # clean windows only
    M={}
    for reg,(X,y) in {"clean":(Xc,yc),"aug":(np.vstack([Xc,Xa]),np.r_[yc,ya])}.items():
        Xs=scl.transform(X)
        M[f"ridge_{reg}"]=Ridge(alpha=100.0).fit(Xs,y)
        M[f"hgb_{reg}"]=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=150,max_leaf_nodes=15,max_depth=None,min_samples_leaf=20,l2_regularization=1.0,early_stopping=False,random_state=seed).fit(Xs,y)
    zero=[SENS[j] for j in range(14) if not (np.isfinite(SD[j]) and SD[j]>1e-12)]
    return dict(MU=MU,SD=SD,scl=scl,M=M,units=list(units),aug_meta=meta,zero_sd=zero)
def observe(B,w,sc,on,pos):
    x=apply(w,sc,on,pos,B["SD"]); fi=fault_indicator(x); X=B["scl"].transform(feats(impute(x,B["MU"]),B["MU"],B["SD"])[None])[0]
    P={m:float(B["M"][m].predict(X[None])[0]) for m in B["M"]}; return X,P,fi
def trunc_rows(ds,seed,sp,stream,df,units,B,scen_list,ns,fold=None):
    R=[]
    for u in units:
        a=df[df.unit==u][SENS].values; T=len(a); eng=f"{ns}:{u}"
        rng,_=rng_for(ds,seed,sp,"trunc",stream,eng,"na","na"); ends=sorted(rng.choice(np.arange(1,T+1),10,replace=False).tolist())
        for d,e in enumerate(ends):
            w=window(a,e)
            for sc in scen_list:
                on,pos,k=(None,None,None) if sc=="clean" else draw(ds,seed,sp,stream,eng,e,parse(sc)[1])
                X,P,fi=observe(B,w,sc,on,pos)
                R.append(dict(engine=eng,fold=fold,draw=d,wend=e,scen=sc,y_raw=float(T-e),onset=on,pos=pos,key=k,fi=fi,X=X,**{f"raw_{m}":P[m] for m in P}))
    return R
HEADS=["H_log","H_sq","pred_only","fault_indicator"]
def fit_heads(oof,seed,pm):
    X=np.array([r["X"] for r in oof]); p=np.maximum(np.array([r[f"raw_{pm}"] for r in oof]),0); y=np.array([r["y_raw"] for r in oof]); Z=np.c_[X,p]
    g=lambda: GradientBoostingRegressor(loss="squared_error",n_estimators=100,max_depth=2,learning_rate=.05,min_samples_leaf=20,subsample=1.0,random_state=seed)
    return {"H_log":g().fit(Z,np.log1p(np.abs(y-p))),"H_sq":g().fit(Z,(y-p)**2/CAP**2),"pred_only":g().fit(p[:,None],(y-p)**2/CAP**2)}
def score(H,pm,rows):
    X=np.array([r["X"] for r in rows]); p=np.maximum(np.array([r[f"raw_{pm}"] for r in rows]),0); Z=np.c_[X,p]
    return {"H_log":H["H_log"].predict(Z),"H_sq":H["H_sq"].predict(Z),"pred_only":H["pred_only"].predict(p[:,None]),"fault_indicator":np.array([r["fi"] for r in rows])}
def order(s,engines,draws):  # ascending risk; ties by SHA256(engine) then draw position (no labels)
    return np.lexsort((np.asarray(draws),np.array([engine_order_key(e) for e in engines]),np.asarray(s)))
def metrics(p,t):
    e=p-t; n=len(t)
    if n==0: return dict(n=0,RMSE=None,MAE=None,MSE=None,ME=None,frac0=None,over20=None,under20=None)
    return dict(n=n,RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),MSE=float(np.mean(e**2)),ME=float(e.mean()),frac0=float(np.mean(p==0)),over20=float(np.mean(e>20)),under20=float(np.mean(e<-20)))
def cell(p,t,s,thr,engines,draws):
    out=dict(full=metrics(p,t)); acc=s<=thr; out["thr"]=dict(thr=float(thr),k=int(acc.sum()),ratio=float(acc.mean()),**{f"ret_{a}":b for a,b in metrics(p[acc],t[acc]).items()})
    o=order(s,engines,draws); oo=np.argsort(np.abs(p-t),kind="stable")
    for k in C["eval"]["fixed_k"]: out[f"k{k}"]=dict(sel=metrics(p[o[:k]],t[o[:k]]),oracle=metrics(p[oo[:k]],t[oo[:k]]))
    return out
from scipy.stats import spearmanr
from . import bootstrap as BS
def strata_metrics(p,t):
    return {f"y_le125":metrics(p[t<=CAP],t[t<=CAP]),"y_gt125":metrics(p[t>CAP],t[t>CAP])}
def anchor(sf,sc,ef,ec):
    """same-engine clean anchor; per-engine deltas. NA when constant/uncomputable."""
    dr=sf-sc; de=np.abs(ef)-np.abs(ec); nz=(dr!=0)&(de!=0)
    agree=float(np.mean(np.sign(dr[nz])==np.sign(de[nz]))) if nz.any() else None
    rho=None if (np.ptp(dr)==0 or np.ptp(de)==0) else float(spearmanr(dr,de).statistic)
    return dict(dir_agree=agree,n_nonzero=int(nz.sum()),spearman=rho,const_all_increase=float(np.mean(de>0)),per_engine_agree=np.where(nz,(np.sign(dr)==np.sign(de)).astype(float),np.nan))
def flat(d,pre=""):
    o={}
    for k,v in d.items():
        if isinstance(v,dict): o.update(flat(v,f"{pre}{k}."))
        elif not isinstance(v,np.ndarray): o[f"{pre}{k}"]=v
    return o
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="../results")
    ap.add_argument("--data-root",default=None,help="C-MAPSS raw data directory")
    ap.add_argument("--ds",nargs="*"); ap.add_argument("--seeds",nargs="*",type=int); ap.add_argument("--streams",nargs="*",type=int); ap.add_argument("--reps",type=int)
    a=ap.parse_args()
    if a.data_root: DATA.set_root(a.data_root)
    DSS,SEEDS,TST,REPS=C["data"]["datasets"],C["split"]["train_seeds"],ST["test"],C["eval"]["bootstrap"]["reps"]
    DSS,SEEDS,TST,REPS=a.ds or DSS,a.seeds or SEEDS,a.streams or TST,a.reps or REPS
    os.makedirs(a.out,exist_ok=True); t0=time.time(); BUDGET=C["budget_phase2_seconds"]
    json.dump(dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),data_root=DATA.ROOT,datasets=DSS,seeds=SEEDS,streams=TST,reps=REPS,subset_run=bool(a.ds or a.seeds or a.streams or a.reps)),open(f"{a.out}/run_info.json","w"),indent=1)
    BOOTCELLS={}; MET=[]; ANC=[]
    with threadpool_limits(1):
        # ---- stage 1: main fits + full per-sample tables (priority 1) ----
        for ds in DSS:
            tr,te,rul=load(ds)
            for seed in SEEDS:
                tag=f"{ds}_s{seed}"; S=split(seed); json.dump(S,open(f"{a.out}/split_{tag}.json","w")); oof=[]; bounds=[]
                for j,fu in enumerate(folds(S["train70"])):
                    B=fit_bundle(ds,seed,tr,[u for u in S["train70"] if u not in fu],f"fold{j}")
                    assert not set(fu)&set(B["units"])
                    bounds.append(dict(split=f"fold{j}",train_units=B["units"],heldout_units=fu,MU=B["MU"].tolist(),SD=B["SD"].tolist(),zero_sd=B["zero_sd"],scaler_mean=B["scl"].mean_.tolist(),scaler_scale=B["scl"].scale_.tolist(),n_aug=len(B["aug_meta"])))
                    pd.DataFrame(B["aug_meta"],columns=["engine","wend","scen","onset","pos"]).assign(split=f"fold{j}").to_csv(f"{a.out}/augfaults_{tag}_fold{j}.csv",index=False)
                    oof+=trunc_rows(ds,seed,f"fold{j}",ST["oof"],tr,fu,B,TRAIN_SCEN,"train",fold=j)
                F=fit_bundle(ds,seed,tr,S["train70"],"final")
                bounds.append(dict(split="final",train_units=F["units"],MU=F["MU"].tolist(),SD=F["SD"].tolist(),zero_sd=F["zero_sd"],scaler_mean=F["scl"].mean_.tolist(),scaler_scale=F["scl"].scale_.tolist(),n_aug=len(F["aug_meta"])))
                pd.DataFrame(F["aug_meta"],columns=["engine","wend","scen","onset","pos"]).assign(split="final").to_csv(f"{a.out}/augfaults_{tag}_final.csv",index=False)
                json.dump(bounds,open(f"{a.out}/fit_boundaries_{tag}.json","w"))
                cal=trunc_rows(ds,seed,"final",ST["calib"],tr,S["calib15"],F,TRAIN_SCEN,"train")
                tests={}
                for st in TST:
                    R=[]
                    for u in range(1,101):
                        a2=te[te.unit==u][SENS].values; e=len(a2); w=window(a2,e); eng=f"test:{u}"
                        for sc in TEST_SCEN:
                            on,pos,k=(None,None,None) if sc=="clean" else draw(ds,seed,"final",st,eng,e,parse(sc)[1])
                            X,P,fi=observe(F,w,sc,on,pos); R.append(dict(engine=eng,draw=0,wend=e,scen=sc,y_raw=float(rul[u-1]),onset=on,pos=pos,key=k,fi=fi,X=X,**{f"raw_{m}":P[m] for m in P}))
                    tests[st]=R
                def table(rows,extra):
                    df=pd.DataFrame([{k:v for k,v in r.items() if k!="X"} for r in rows]); Xd=pd.DataFrame(np.array([r["X"] for r in rows]),columns=[f"x{i}" for i in range(56)])
                    for m in F["M"]: df[f"pred_{m}"]=np.maximum(df[f"raw_{m}"],0)
                    for k,v in extra.items(): df[k]=v
                    return pd.concat([df,Xd],axis=1)
                oofdf=table(oof,{}); oofdf["fold_train_units"]=[json.dumps(bounds[f]["train_units"]) for f in oofdf.fold]
                assert all(int(r.split(":")[1]) not in json.loads(l) for r,l in zip(oofdf.engine,oofdf.fold_train_units))  # no self-engine
                calx={}; testx={st:{} for st in TST}; thrs={}
                for pm in F["M"]:
                    H=fit_heads(oof,seed,pm); sc_cal=score(H,pm,cal)
                    thr={h:float(np.quantile(sc_cal[h],0.8,method="linear")) for h in HEADS}; thrs[pm]=thr
                    for h in HEADS: calx[f"score_{pm}_{h}"]=sc_cal[h]; calx[f"acc_{pm}_{h}"]=(sc_cal[h]<=thr[h]).astype(int)
                    for st,R in tests.items():
                        SC=score(H,pm,R); eng_all=np.array([r["engine"] for r in R]); scen_all=np.array([r["scen"] for r in R])
                        for h in HEADS: testx[st][f"score_{pm}_{h}"]=SC[h]; testx[st][f"acc_{pm}_{h}"]=(SC[h]<=thr[h]).astype(int); testx[st][f"sel80_{pm}_{h}"]=np.zeros(len(R),int)
                        ci=np.where(scen_all=="clean")[0]; pc=np.maximum([R[i][f"raw_{pm}"] for i in ci],0); tc=np.array([R[i]["y_raw"] for i in ci])
                        for scen in TEST_SCEN:
                            idx=np.where(scen_all==scen)[0]; assert list(eng_all[idx])==[f"test:{u}" for u in range(1,101)]
                            p=np.maximum([R[i][f"raw_{pm}"] for i in idx],0); t=np.array([R[i]["y_raw"] for i in idx]); en=list(eng_all[idx])
                            for h in HEADS:
                                s=SC[h][idx]; c=cell(p,t,s,thr[h],en,np.zeros(len(idx))); c["strata_full"]=strata_metrics(p,t)
                                o80=order(s,en,np.zeros(len(idx)))[:80]; testx[st][f"sel80_{pm}_{h}"][idx[o80]]=1
                                c["strata_sel80"]=strata_metrics(p[o80],t[o80])
                                MET.append(dict(ds=ds,seed=seed,stream=st,pred=pm,head=h,scen=scen,**flat(c)))
                                if scen!="clean":
                                    A_=anchor(s,SC[h][ci],p-t,pc-tc); ANC.append(dict(ds=ds,seed=seed,stream=st,pred=pm,head=h,scen=scen,**{k:v for k,v in A_.items() if k!="per_engine_agree"}))
                                    BOOTCELLS.setdefault((ds,h,scen),{}).setdefault((seed,st),{})[pm]=dict(p=p,t=t,s=s,sha=np.array([engine_order_key(x) for x in en]),eng_idx=np.arange(100))
                                    BOOTCELLS.setdefault((ds,h,"clean"),{}).setdefault((seed,st),{})[pm]=dict(p=pc,t=tc,s=SC[h][ci],sha=np.array([engine_order_key(f"test:{u}") for u in range(1,101)]),eng_idx=np.arange(100))
                oofdf.to_csv(f"{a.out}/oof_{tag}.csv",index=False); table(cal,calx).to_csv(f"{a.out}/calib_{tag}.csv",index=False)
                for st in TST: table(tests[st],testx[st]).to_csv(f"{a.out}/test_{tag}_st{st}.csv",index=False)
                json.dump(thrs,open(f"{a.out}/thresholds_{tag}.json","w"),indent=1)
                pd.DataFrame(MET).to_csv(f"{a.out}/metrics_long.csv",index=False); pd.DataFrame(ANC).to_csv(f"{a.out}/anchor.csv",index=False)
                print(tag,"stage1 done",round(time.time()-t0),"s",flush=True)
        # equal-weight average of metrics over seed x stream combos (never of predictions/scores)
        M=pd.DataFrame(MET); num=[c for c in M.columns if c not in ("ds","seed","stream","pred","head","scen") and pd.api.types.is_numeric_dtype(M[c])]
        M.groupby(["ds","pred","head","scen"])[num].mean().reset_index().to_csv(f"{a.out}/metrics_avg_combos.csv",index=False)
        # ---- stage 2: pre-specified main-contrast bootstrap intervals (priority 2) ----
        rows=[]; status="complete"; preds=list(F["M"])
        for (ds,h,scen),cells in sorted(BOOTCELLS.items()):
            if scen=="clean" or h not in ("H_log","H_sq"): continue
            if time.time()-t0>BUDGET: status="incomplete_timeout"; break
            reps=BS.run(cells,BOOTCELLS[(ds,h,"clean")],preds,reps=REPS,seed=C["eval"]["bootstrap"]["seed"])
            arr={pm:np.array([r[pm] for r in reps]) for pm in preds}
            for pm in preds:
                for q,nm in enumerate(["selection_regret","base_floor","total"]): rows.append(dict(ds=ds,head=h,scen=scen,contrast=pm,quantity=nm,lo=np.percentile(arr[pm][:,q],2.5),hi=np.percentile(arr[pm][:,q],97.5)))
            for i,p1 in enumerate(preds):
                for p2 in preds[i+1:]:
                    for q,nm in enumerate(["selection_regret","base_floor","total"]):
                        d=arr[p1][:,q]-arr[p2][:,q]; rows.append(dict(ds=ds,head=h,scen=scen,contrast=f"{p1}-{p2}",quantity=nm,lo=np.percentile(d,2.5),hi=np.percentile(d,97.5)))
        pd.DataFrame(rows).assign(status=status,reps=REPS).to_csv(f"{a.out}/bootstrap_intervals.csv",index=False)
    json.dump(dict(end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),elapsed_s=round(time.time()-t0,1),bootstrap_status=status),open(f"{a.out}/run_end.json","w"),indent=1)
    print("done",status,round(time.time()-t0),"s",flush=True)
    if status != "complete": raise SystemExit(2)
if __name__=="__main__": main()
