#!/bin/sh
set -eu
data_root="$1"
output_root="$2"
for dataset in FD001 FD003; do
  for seed in 42 43 44; do
    python -m riskselrul reproduce --data-root "$data_root" --dataset "$dataset" --seed "$seed" --streams 9001 9002 9003 --bootstrap-reps 2000 --out "$output_root/${dataset}_s${seed}"
  done
done
