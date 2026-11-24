$ErrorActionPreference = 'Continue'
$docker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$root = Split-Path -Parent $PSScriptRoot
$mounts = @('--mount',"type=bind,source=$root,target=/workspace,readonly",'--mount',"type=bind,source=$PSScriptRoot,target=/workspace/full_receipt_20261117")
$record = Join-Path $PSScriptRoot ('records\round2_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory $record | Out-Null
$full = Get-Content (Join-Path $PSScriptRoot 'evaluation/full_val_v1/summary.json') -Raw | ConvertFrom-Json
if ($full.status -ne 'completed' -or $full.n -ne 154) { throw 'Full validation must finish first' }
foreach ($stage in @('smoke','train','export','validate')) {
    & $docker run --rm --gpus all --shm-size 2g @mounts mogumogu-korie-training:20261117 python /workspace/full_receipt_20261117/train_round2_v2.py $stage --run /workspace/full_receipt_20261117/round2/run_001 --repo /workspace/full_receipt_20261117/PaddleOCR --data-root /workspace/full_receipt_20261117/data/ocr 2>&1 | Out-File (Join-Path $record ($stage + '.log')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage; see $record" }
}
Write-Output "Round2 validation completed: $record. Review validation before freezing test."
