param([Parameter(Mandatory=$true)][string]$DataRoot, [Parameter(Mandatory=$true)][string]$Out)
$ErrorActionPreference = 'Stop'
foreach ($dataset in @('FD001','FD003')) {
    foreach ($seed in @(42,43,44)) {
        $runPath = Join-Path $Out "${dataset}_s${seed}"
        python -m riskselrul reproduce --data-root $DataRoot --dataset $dataset --seed $seed --streams 9001 9002 9003 --bootstrap-reps 2000 --out $runPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
