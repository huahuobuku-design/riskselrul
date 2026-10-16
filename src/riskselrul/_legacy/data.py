import numpy as np, pandas as pd, hashlib, os
from .faults import SENS
COLS=["unit","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
ROOT=os.environ.get("WP05A_DATA_ROOT",os.path.join(os.path.dirname(__file__),"..","data","raw"))
def set_root(p):
    global ROOT; ROOT=p
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def load(ds):
    tr=pd.read_csv(f"{ROOT}/train_{ds}.txt",sep=r"\s+",header=None,names=COLS)
    te=pd.read_csv(f"{ROOT}/test_{ds}.txt",sep=r"\s+",header=None,names=COLS)
    rul=np.loadtxt(f"{ROOT}/RUL_{ds}.txt")
    return tr,te,rul
def split(seed):
    p=np.random.default_rng(seed).permutation(np.arange(1,101)); return dict(train70=p[:70].tolist(),val15=p[70:85].tolist(),calib15=p[85:].tolist())
def folds(train70): return [train70[14*j:14*j+14] for j in range(5)]
def window(a,e,W=30):
    w=a[max(0,e-W):e]; return w if len(w)==W else np.vstack([np.repeat(w[:1],W-len(w),0),w])
def sensor_stats(tr,units):
    v=tr[tr.unit.isin(units)][SENS]; return v.mean().values, v.std(ddof=1).values
T_AX=np.arange(30)-14.5
def feats(w,MU,SD):
    sd=np.where(np.isfinite(SD)&(SD>1e-12),SD,1.0); z=(w-MU)/sd
    return np.concatenate([z.mean(0),z.std(0),z[-1],(T_AX@(z-z.mean(0)))/(T_AX@T_AX)])
