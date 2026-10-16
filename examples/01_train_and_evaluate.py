"""Example 1 - train predictors and risk heads, then evaluate the fault streams.

Runs a reduced grid by default so it finishes quickly. Drop ``--datasets``,
``--seeds`` and ``--streams`` overrides to reproduce the released grid.

    python examples/01_train_and_evaluate.py --out runs/example01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riskselrul.data import cmapss  # noqa: E402
from riskselrul.evaluation import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="directory with train_/test_/RUL_ files")
    ap.add_argument("--out", default="runs/example01")
    ap.add_argument("--datasets", nargs="*", default=["FD001"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--streams", nargs="*", type=int, default=[9001])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--budget", type=float, default=600.0)
    a = ap.parse_args()

    if a.data_root:
        cmapss.set_root(a.data_root)

    summary = pipeline.run(datasets=a.datasets, seeds=a.seeds, streams=a.streams,
                           reps=a.reps, out=a.out, budget=a.budget)
    print("summary:", summary)
    print("per-sample tables and metrics written under", a.out)


if __name__ == "__main__":
    main()
