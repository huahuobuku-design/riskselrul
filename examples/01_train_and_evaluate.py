"""Run from installed package; real NASA files must be supplied explicitly."""
import argparse
from riskselrul.pipeline import reproduce

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
reproduce(args.data_root, args.out, dataset='FD001', seed=42, streams=(9001,), bootstrap_reps=2000)
