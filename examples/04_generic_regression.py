"""Part C: non-RUL reuse demo of riskselrul 0.1.0 public array API (no library changes).
Run after installing .[train]: python examples/04_generic_regression.py --out generic_example"""
import sys, json, os, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import riskselrul
from riskselrul import calibrate_threshold, fixed_k_metrics, prediction_metrics, risk_coverage_curve
parser = argparse.ArgumentParser(description="Run reproducible general-regression examples with the public array API")
parser.add_argument("--out", required=True, type=Path, help="A new output directory")
args = parser.parse_args()
if args.out.exists():
    parser.error("Output directory already exists; choose a new directory")
args.out.mkdir(parents=True)
OUT = str(args.out)
def data():
    d=load_diabetes(scaled=False); yield "diabetes_real",d.data,d.target
    X,y=make_friedman1(n_samples=1000,n_features=10,noise=1.0,random_state=20261017); yield "friedman1_SYNTHETIC",X,y
def aurc(p,y,s,ids):
    c=risk_coverage_curve(p,y,s,ids); return float(np.mean([r["retained_mse"] for r in c]))
summary=[]
for name,X,y in data():
  n=len(y)
  for seed in (42,43,44):
    rng=np.random.default_rng(seed); perm=rng.permutation(n)
    nf=int(round(.6*n)); nc=int(round(.2*n))
    fit,cal,te=perm[:nf],perm[nf:nf+nc],perm[nf+nc:]
    ids=np.array([f"{name}_{i}" for i in range(n)])
    Xf,yf=X[fit],y[fit]
    kf=KFold(5,shuffle=True,random_state=seed); oof=np.zeros(len(fit)); folds=[]
    for f,(a,b) in enumerate(kf.split(Xf)):
        m=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(Xf[a],yf[a]); oof[b]=m.predict(Xf[b])
        sc=m.named_steps["standardscaler"]
        folds.append(dict(fold=f,train_ids=ids[fit][a].tolist(),val_ids=ids[fit][b].tolist(),scaler_mean=sc.mean_.tolist(),scaler_scale=sc.scale_.tolist()))
    main=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(Xf,yf)
    rt=np.log1p(np.abs(yf-oof))   # OOF residual only
    Zf=np.column_stack([Xf,oof]); rs=StandardScaler().fit(Zf); head=Ridge(alpha=1.0).fit(rs.transform(Zf),rt)
    def scores(idx,seed_off):
        p=main.predict(X[idx]); z=rs.transform(np.column_stack([X[idx],p]))
        return p,dict(risk_head=head.predict(z),random=np.random.default_rng(seed+10000).random(len(idx)) if seed_off is None else seed_off,prediction_only=p)
    pc,sc_c=scores(cal,None); pt,sc_t=scores(te,None)
    # independent random draws for cal and test from the seed+10000 generator
    g=np.random.default_rng(seed+10000); sc_c["random"]=g.random(len(cal)); sc_t["random"]=g.random(len(te))
    yt=y[te]; idt=ids[te]; k=int(np.floor(.8*len(te)))
    rows_c=pd.DataFrame(dict(id=ids[cal],y_true=y[cal],pred=pc,**{f"score_{h}":v for h,v in sc_c.items()}))
    rows_t=pd.DataFrame(dict(id=idt,y_true=yt,pred=pt,**{f"score_{h}":v for h,v in sc_t.items()}))
    rec=dict(dataset=name,seed=seed,n=n,n_fit=len(fit),n_cal=len(cal),n_test=len(te),fixed_k=k,
             full_test=prediction_metrics(pt,yt),thresholds={},results={})
    for h in ("risk_head","random","prediction_only"):
        thr=calibrate_threshold(sc_c[h],0.8); rec["thresholds"][h]=thr
        acc=sc_t[h]<=thr; rows_t[f"acc_{h}"]=acc.astype(int)
        pm=prediction_metrics(pt[acc],yt[acc])
        fk=fixed_k_metrics(pt,yt,sc_t[h],k,idt)
        rec["results"][h]=dict(threshold=dict(n_accepted=int(acc.sum()),acceptance_ratio=float(acc.mean()),rmse=pm["rmse"],mse=pm["mse"]),
            fixed_k=dict(n_accepted=k,rmse=fk["retained"]["rmse"],mse=fk["retained"]["mse"],oracle_mse_posthoc=fk["oracle"]["mse"],random_expected_mse=fk["random_expected_mse"]),
            aurc_mse=aurc(pt,yt,sc_t[h],idt))
    e=np.abs(pt-yt); rec["results"]["oracle_offline"]=dict(aurc_mse=aurc(pt,yt,e,idt),note="post-hoc uses test labels; reference only")
    tag=f"{name}_s{seed}"
    rows_c.to_csv(f"{OUT}/cal_{tag}.csv",index=False); rows_t.to_csv(f"{OUT}/test_{tag}.csv",index=False)
    pd.DataFrame(dict(id=ids[fit],y_true=yf,oof_pred=oof,oof_log1p_absres=rt)).to_csv(f"{OUT}/oof_{tag}.csv",index=False)
    json.dump(dict(split=dict(fit=ids[fit].tolist(),cal=ids[cal].tolist(),test=idt.tolist()),folds=folds,
        main_scaler=dict(mean=main[0].mean_.tolist(),scale=main[0].scale_.tolist()),main_coef=main[1].coef_.tolist(),main_intercept=float(main[1].intercept_),
        risk_scaler=dict(mean=rs.mean_.tolist(),scale=rs.scale_.tolist()),risk_coef=head.coef_.tolist(),risk_intercept=float(head.intercept_)),open(f"{OUT}/records_{tag}.json","w"))
    summary.append(rec)
json.dump(dict(riskselrul_version=riskselrul.__version__,note="software reuse demo; not clinical; friedman1 is synthetic",runs=summary),open(f"{OUT}/summary_C.json","w"),indent=1,allow_nan=False)
for r in summary:
    print(r["dataset"],r["seed"],{h:(v.get("threshold",{}).get("n_accepted"),round(v.get("threshold",{}).get("rmse") or float('nan'),2),round(v["fixed_k"]["rmse"],2) if "fixed_k" in v else None,round(v["aurc_mse"],1)) for h,v in r["results"].items()})
