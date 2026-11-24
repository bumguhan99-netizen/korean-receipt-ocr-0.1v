$ErrorActionPreference = 'Continue'
$docker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$root = Split-Path -Parent $PSScriptRoot
$mounts = @('--mount',"type=bind,source=$root,target=/workspace,readonly",'--mount',"type=bind,source=$PSScriptRoot,target=/workspace/full_receipt_20261117")
$record = Join-Path $PSScriptRoot ('records\final_test_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory $record | Out-Null
function Stage($name,$arguments,$image='mogumogu-korie-training:20261117') {
    & $docker run --rm --gpus all --shm-size 2g @mounts $image python @arguments 2>&1 | Out-File (Join-Path $record ($name + '.log')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Failed $name. See $record" }
}
Stage 'policy' @('/workspace/full_receipt_20261117/finalize_policy_v1.py')
$common=@('--run','/workspace/full_receipt_20261117/round2/run_001','--repo','/workspace/full_receipt_20261117/PaddleOCR','--data-root','/workspace/full_receipt_20261117/data/ocr')
Stage 'freeze' (@('/workspace/full_receipt_20261117/train_round2_v2.py','freeze')+$common)
Stage 'crop_test' (@('/workspace/full_receipt_20261117/train_round2_v2.py','test')+$common)
Stage 'full_test' @('/workspace/full_receipt_20261117/full_receipt_v2.py','--images','/workspace/full_receipt_20261117/data/detection/test/images','--detector','/workspace/full_receipt_20261117/models/PP-OCRv5_mobile_det','--baseline','/workspace/training/exports/selected','--candidate','/workspace/full_receipt_20261117/round2/run_001/export','--output','/workspace/full_receipt_20261117/evaluation/full_test_v1','--frozen','/workspace/full_receipt_20261117/frozen_full_test_v1.json') 'mogumogu-korie-eval:20261117'
Write-Output "Final test finished: $record"
