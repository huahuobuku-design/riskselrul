"""Fault generator. raw window: (30,14) in config sensor order. Pure numpy; deterministic from rng_keys."""
import numpy as np
from .rng_keys import rng_for
SENS=["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
A=["s2","s4","s8","s11","s13","s15","s20"]; B=["s3","s7","s9","s12","s14","s17","s21"]
NCH={"dropout1":1,"dropout3":3,"stuck1":1,"stuck3":3,"bias2":3,"bias4":3,"drift4":3}
TRAIN_SCEN=["clean","A_dropout1","A_dropout3","A_stuck1","A_stuck3","A_bias+2","A_bias+4","A_drift+4"]
TEST_SCEN=["clean"]+[f"{g}_{t}" for g in "AB" for t in ["dropout1","dropout3","stuck1","stuck3"]]+[f"{g}_{t}{s}{m}" for g in "AB" for t,m in [("bias",2),("bias",4),("drift",4)] for s in "+-"]
def parse(scen):
    """'B_bias-4' -> (group,'bias4',type,sign,mag)"""
    if scen=="clean": return None
    g,r=scen.split("_")
    for t in ("dropout","stuck"):
        if r.startswith(t): return g,r,t,1,0
    t="bias" if r.startswith("bias") else "drift"; sign=1 if r[len(t)]=="+" else -1; m=int(r[len(t)+1:])
    return g,f"{t}{m}",t,sign,m
def draw(ds,seed,split,stream,engine,wend,base):
    """onset and within-group positions; key excludes sign and group => pairs share them."""
    rng,k=rng_for(ds,seed,split,"fault",stream,engine,wend,base)
    on=int(rng.integers(5,26)); pos=rng.choice(7,size=NCH[base],replace=False)
    return on,[int(p) for p in pos],k
def apply(raw,scen,on,pos,SD):
    """raw (30,14) float; SD (14,) raw ddof1 SD (0 amplitude if SD invalid). returns window with NaN for dropout."""
    x=np.array(raw,float,copy=True)
    p=parse(scen)
    if p is None: return x
    g,base,t,sign,m=p; grp=A if g=="A" else B
    for q in pos:
        j=SENS.index(grp[q]); sd=SD[j] if (np.isfinite(SD[j]) and SD[j]>1e-12) else 0.0
        if t=="dropout": x[on:,j]=np.nan
        elif t=="stuck": x[on:,j]=x[on,j]
        elif t=="bias": x[on:,j]+=sign*m*sd
        elif t=="drift": x[on:,j]+=sign*m*sd*np.linspace(0,1,30-on)
    return x
def impute(x,MU):
    y=x.copy(); r,c=np.where(np.isnan(y)); y[r,c]=np.asarray(MU)[c]; return y
def fault_indicator(x):
    """pre-imputation window (30,14): mean NaN frac + mean runf (longest equal run L>=5 -> L/30)."""
    nanf=np.isnan(x).mean(0); runf=np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        L=cur=1
        for t in range(1,x.shape[0]):
            cur=cur+1 if x[t,j]==x[t-1,j] else 1; L=max(L,cur)
        runf[j]=L/30 if L>=5 else 0.0
    return float(nanf.mean()+runf.mean())
def aug_choice(ds,seed,split,engine,wend,stream=2026101601):
    rng,k=rng_for(ds,seed,split,"aug_choice",stream,engine,wend,"na")
    return TRAIN_SCEN[1+int(rng.integers(0,7))],k
