"""Paired engine-cluster bootstrap (frozen; executed only in phase 2).
cells: dict combo(seed,stream)-> dict pred-> dict(p,t,s,eng_idx (0..99 test unit-1), sha (engine order key)). Fixed models/thresholds.
Per replicate: common indices idx (100 draws w/ replacement of engines); for each combo & predictor recompute fixed-k selection,
its own oracle, full MSE; average 9 combos equally; then pairwise predictor differences. Duplicates kept; ties by (score, sha, draw position)."""
import numpy as np
def k_readout(p,t,s,sha,k=80):
    e2=(p-t)**2; dpos=np.arange(len(p))
    o=np.lexsort((dpos,sha,s))[:k]; oo=np.lexsort((dpos,np.abs(p-t)))[:k]
    return dict(sel=e2[o].mean(),oracle=e2[oo].mean(),full=e2.mean())
def resample(cell,idx):
    assert np.array_equal(np.asarray(cell["eng_idx"]),np.arange(100))  # test cell: one terminal row per engine, unit order
    pos=np.asarray(idx)
    return {k:np.asarray(v)[pos] for k,v in cell.items()}
def run(cells_fault,cells_clean,preds,reps=2000,seed=20261016,k=80):
    rng=np.random.default_rng(seed); out=[]
    for b in range(reps):
        idx=rng.integers(0,100,100); row={}
        for pm in preds:
            acc=[]
            for c in cells_fault:
                f=k_readout(**{x:resample(cells_fault[c][pm],idx)[x] for x in ("p","t","s","sha")},k=k)
                cl=resample(cells_clean[c][pm],idx); fc=np.mean((cl["p"]-cl["t"])**2)
                acc.append((f["sel"]-f["oracle"],f["oracle"]-fc,f["sel"]-fc))
            row[pm]=np.mean(acc,0)
        out.append(row)
    return out
