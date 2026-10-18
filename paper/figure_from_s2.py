"""Generate publication figures and expand the local manuscript from verified tables."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--s2",type=Path,required=True,help="Extracted Supplementary S2 folder")
parser.add_argument("--out",type=Path,required=True,help="New figure output folder")
args=parser.parse_args()
HERE=args.out
HERE.mkdir(parents=True,exist_ok=False)
DATA=args.s2/"expanded"
(HERE/"figures").mkdir(exist_ok=True)
C=json.loads((DATA/"out/C/summary_C.json").read_text())
tab=[]
for dataset in ("diabetes_real","friedman1_SYNTHETIC"):
    runs=[r for r in C["runs"] if r["dataset"]==dataset]
    for head in ("risk_head","random","prediction_only"):
        tab.append(dict(dataset=dataset,head=head,
            aurc=np.mean([r["results"][head]["aurc_mse"] for r in runs]),
            acceptance=np.mean([r["results"][head]["threshold"]["acceptance_ratio"] for r in runs]),
            rmse=np.mean([r["results"][head]["threshold"]["rmse"] for r in runs])))
pd.DataFrame(tab).to_csv(HERE/"non_rul_example_summary.csv",index=False)
df=pd.read_csv(DATA/"out/B/B_equal_average.csv",float_precision="round_trip")
ci=pd.read_csv(DATA/"out/B/B_bootstrap_ci_representative.csv",float_precision="round_trip")
reps=["clean","A_bias+4","B_bias+4"]
heads=["H_log","H_sq","pred_only","fault_indicator"]
colors=["#0072B2","#D55E00","#009E73","#CC79A7"]
styles=["-","--","-.",":"]
fig,axes=plt.subplots(2,3,figsize=(7.2,5.8))
for row,ds in enumerate(("FD001","FD003")):
    for col,scenario in enumerate(reps):
        ax=axes[row,col]; curves={h:[] for h in heads}; oracle=[]; random=[]
        for path in sorted((DATA/"out/A").glob(f"{ds}_s*/test_*_st*.csv")):
            frame=pd.read_csv(path,float_precision="round_trip")
            frame=frame[frame.scen==scenario].sort_values("engine",key=lambda x:x.str.split(":").str[1].astype(int))
            errors=(frame.pred_ridge_aug.to_numpy()-frame.y_raw.to_numpy())**2
            keys=np.array([hashlib.sha256(v.encode()).hexdigest() for v in frame.engine])
            for h in heads:
                order=np.lexsort((np.arange(100),keys,frame[f"score_ridge_aug_{h}"]))
                curves[h].append(np.cumsum(errors[order])/np.arange(1,101))
            oracle.append(np.cumsum(np.sort(errors))/np.arange(1,101))
            random.append(errors.mean())
        x=np.arange(1,101)/100
        for h,c,ls in zip(heads,colors,styles):
            ax.plot(x,np.mean(curves[h],axis=0),label=h,color=c,linestyle=ls,lw=1.6)
        ax.axhline(np.mean(random),label="Random expected",color="#555555",lw=1,linestyle="--")
        ax.plot(x,np.mean(oracle,axis=0),label="Oracle (post hoc)",color="black",lw=1)
        ax.set_title(f"{ds}  {scenario}",fontsize=10)
        ax.set_xlim(0,1); ax.tick_params(labelsize=9)
        ax.grid(alpha=.18)
        ax.set_xlabel("Retained fraction",fontsize=10)
        ax.set_ylabel("MSE (cycles²)",fontsize=10)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0,3))
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9,frameon=False)
fig.subplots_adjust(hspace=.65,wspace=.48,bottom=.2,top=.94)
for ext in ("png","pdf","svg"):
    fig.savefig(HERE/f"figures/expanded_ranking.{ext}",dpi=300)
plt.close(fig)

