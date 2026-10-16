"""Public command line; heavy dependencies are loaded only for training/audit."""
import argparse
import csv
import json
from pathlib import Path
import sys


def _write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    from . import __version__
    parser = argparse.ArgumentParser(prog="riskselrul", description="Reproducible selective RUL tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-data", help="Check C-MAPSS format and hashes")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--dataset", choices=("FD001", "FD003"), default="FD001")
    inspect.add_argument("--strict-hashes", action="store_true")
    extract = commands.add_parser("extract-data", help="Extract FD001/FD003 from a supplied NASA ZIP")
    extract.add_argument("--zip", required=True)
    extract.add_argument("--out", required=True)
    run = commands.add_parser("reproduce", help="Train from raw data, evaluate, bootstrap, independently audit")
    run.add_argument("--data-root", required=True)
    run.add_argument("--dataset", choices=("FD001", "FD003"), default="FD001")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--streams", nargs="+", type=int, default=[9001])
    run.add_argument("--bootstrap-reps", type=int, default=2000)
    run.add_argument("--out", required=True)
    audit = commands.add_parser("audit", help="Recompute numeric results without fitting models")
    audit.add_argument("--run", required=True)
    commands.add_parser("list-states", help="List the packaged WP04A inference states")
    predict = commands.add_parser("predict", help="NumPy-only raw-window inference")
    predict.add_argument("--state", required=True)
    predict.add_argument("--windows", required=True, help="Numeric .npy of shape (n,30,21)")
    predict.add_argument("--condition", default="C2")
    predict.add_argument("--acceptance-rate", type=float, default=.8)
    predict.add_argument("--out", required=True)
    curve = commands.add_parser("curve", help="Retrospective fixed-k diagnostic curve from a CSV")
    curve.add_argument("--table", required=True)
    curve.add_argument("--prediction", required=True)
    curve.add_argument("--risk", required=True)
    curve.add_argument("--target", default="y_raw")
    curve.add_argument("--scenario", default="clean", help="Select one scenario; never pool different fault burdens")
    curve.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-data":
            from .data import inspect_data
            print(json.dumps(inspect_data(args.data_root, args.dataset, args.strict_hashes), indent=2))
        elif args.command == "extract-data":
            from .data import extract_cmapss
            print(json.dumps(extract_cmapss(args.zip, args.out), indent=2))
        elif args.command == "reproduce":
            from .pipeline import reproduce
            reproduce(args.data_root, args.out, args.dataset, args.seed, args.streams, args.bootstrap_reps)
        elif args.command == "audit":
            from .audit import audit_run
            report, _ = audit_run(args.run)
            print(json.dumps(report, indent=2, allow_nan=False))
            return int(bool(report["mismatches"]))
        elif args.command == "list-states":
            from .deploy import available_states
            print("\n".join(available_states()))
        elif args.command == "predict":
            import numpy as np
            from .deploy import decide, load_state
            prediction, risk, accepted = decide(np.load(args.windows, allow_pickle=False), load_state(args.state), args.condition, args.acceptance_rate)
            _write_rows(args.out, [dict(index=i, predicted_rul=float(p), risk_score=float(r), accepted=bool(a))
                                  for i, (p, r, a) in enumerate(zip(prediction, risk, accepted))])
        elif args.command == "curve":
            from .metrics import risk_coverage_curve
            with Path(args.table).open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            if rows and "scen" in rows[0]:
                rows = [r for r in rows if r["scen"] == args.scenario]
            if not rows:
                raise ValueError("No rows for this scenario")
            ids = [r.get("engine", str(i)) for i, r in enumerate(rows)]
            if len(set(ids)) != len(ids):
                raise ValueError("Curve requires one row per engine; select one dataset/seed/stream/scenario first")
            _write_rows(args.out, risk_coverage_curve([float(r[args.prediction]) for r in rows],
                [float(r[args.target]) for r in rows], [float(r[args.risk]) for r in rows], ids))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
