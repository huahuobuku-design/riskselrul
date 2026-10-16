# One-shot re-run of the released experiment grid (Windows / PowerShell).
#
#   .\scripts\reproduce_all.ps1
#   $env:BUDGET = "600"; .\scripts\reproduce_all.ps1   # stop stage 2 after 600 s
#
# Requires the C-MAPSS text files under data\raw\ (see scripts\fetch_cmapss.md).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$budget = if ($env:BUDGET) { $env:BUDGET } else { "7200" }
$out = if ($env:OUT) { $env:OUT } else { "runs" }

python -m pip install -e .
python -m unittest discover -s tests -v

python -m riskselrul train --datasets FD001 FD003 --seeds 42 43 44 --out $out --budget $budget
python -m riskselrul audit --run $out

Write-Host "reproduce_all: done -> $out"
