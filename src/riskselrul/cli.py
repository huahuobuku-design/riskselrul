"""Command-line interface.

Subcommands::

    riskselrul train     fit predictors and risk heads, write per-sample tables
    riskselrul audit     recompute metrics from saved per-sample tables
    riskselrul deploy    apply a lean state to raw windows

``train`` runs stage 1 (training + evaluation tables) and then stage 2 (paired
engine-cluster bootstrap) as far as the wall-clock budget allows.
"""

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="riskselrul",
                                 description="Risk-aware selective RUL prediction on C-MAPSS")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit predictors and risk heads on a dataset/seed grid")
    t.add_argument("--datasets", nargs="*", default=None, help="e.g. FD001 FD003")
    t.add_argument("--seeds", nargs="*", type=int, default=None)
    t.add_argument("--streams", nargs="*", type=int, default=None)
    t.add_argument("--reps", type=int, default=None, help="bootstrap replicates")
    t.add_argument("--budget", type=float, default=None, help="wall-clock budget in seconds")
    t.add_argument("--out", default="runs")
    t.add_argument("--data-root", default=None, help="directory holding train_/test_/RUL_ files")

    a = sub.add_parser("audit", help="recompute metrics from saved per-sample tables")
    a.add_argument("--run", default="runs")

    d = sub.add_parser("deploy", help="apply a lean state to a raw-window array")
    d.add_argument("--state", required=True, help="lean state .npz")
    d.add_argument("--windows", help="optional .npy of shape (n, 30, 21)")
    d.add_argument("--condition", default="C2")
    d.add_argument("--coverage", type=float, default=0.8)

    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .data import cmapss
        from .evaluation import pipeline
        if args.data_root:
            cmapss.set_root(args.data_root)
        pipeline.run(datasets=args.datasets, seeds=args.seeds, streams=args.streams,
                     reps=args.reps, out=args.out, budget=args.budget)
        return 0

    if args.cmd == "audit":
        from .evaluation import audit
        res = audit.verify(args.run)
        return 1 if res["mismatches"] else 0

    if args.cmd == "deploy":
        import numpy as np
        from .deploy import lean
        state = lean.load(args.state)
        if not args.windows:
            print("state loaded; pass --windows to run inference")
            return 0
        raw = np.load(args.windows)
        p, q, acc = lean.decide(raw, state, args.condition, args.coverage)
        print("n=%d accepted=%d mean_pred=%.4f mean_risk=%.6f"
              % (len(p), int(acc.sum()), float(p.mean()), float(q.mean())))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
