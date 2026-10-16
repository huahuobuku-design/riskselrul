"""Example 2 - coverage / error trade-off from a completed run.

Reads ``metrics_long.csv`` produced by example 1 or by ``riskselrul train`` and
writes a small PNG.

    python examples/02_coverage_error_curve.py --run runs/example01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/example01")
    ap.add_argument("--out", default="coverage_error.png")
    ap.add_argument("--head", default="H_log")
    ap.add_argument("--pred", default="hgb_clean")
    a = ap.parse_args()

    M = pd.read_csv(os.path.join(a.run, "metrics_long.csv"))
    sub = M[(M["head"] == a.head) & (M.pred == a.pred)]
    if sub.empty:
        raise SystemExit("no rows for head=%s pred=%s in %s" % (a.head, a.pred, a.run))

    fig, ax = plt.subplots(figsize=(6, 4))
    for scen, g in sub.groupby("scen"):
        g = g.sort_values("thr.ratio")
        ax.plot(g["thr.ratio"], g["thr.ret_RMSE"], marker=".", linewidth=1, label=scen)
    ax.set_xlabel("accepted fraction")
    ax.set_ylabel("retained RMSE")
    ax.set_title("coverage / error trade-off  (%s, %s)" % (a.head, a.pred))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
