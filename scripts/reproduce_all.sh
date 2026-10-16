#!/usr/bin/env bash
# One-shot re-run of the released experiment grid.
#
#   bash scripts/reproduce_all.sh            # full grid
#   BUDGET=600 bash scripts/reproduce_all.sh # stop stage 2 after 600 s
#
# Requires the C-MAPSS text files under data/raw/ (see scripts/fetch_cmapss.md).
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET="${BUDGET:-7200}"
OUT="${OUT:-runs}"

python -m pip install -e .
python -m unittest discover -s tests -v

riskselrul train --datasets FD001 FD003 --seeds 42 43 44 --out "$OUT" --budget "$BUDGET"
riskselrul audit --run "$OUT"

echo "reproduce_all: done -> $OUT"
