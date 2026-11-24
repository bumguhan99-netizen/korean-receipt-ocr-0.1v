$ErrorActionPreference = 'Stop'
$docker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$root = Split-Path -Parent $PSScriptRoot
$record = Join-Path $PSScriptRoot ('records\validation_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory $record | Out-Null
$mounts = @('--mount',"type=bind,source=$root,target=/workspace,readonly",'--mount',"type=bind,source=$PSScriptRoot,target=/workspace/full_receipt_20261117")
function Run-Stage($name, $arguments) {
    $ErrorActionPreference = 'Continue'
    & $docker @arguments 2>&1 | Out-File (Join-Path $record ($name + '.log')) -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $name; see $record" }
}
$base = @('run','--rm','--gpus','all','--shm-size','2g') + $mounts + @('mogumogu-korie-eval:20261117','python')
Run-Stage 'gpu' ($base + @('/workspace/training/gpu_check.py'))
Run-Stage 'prepare' ($base + @('/workspace/training/korie.py','prepare','--data-root','/workspace/full_receipt_20261117/data/ocr','--dictionary','/workspace/full_receipt_20261117/PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt','--reference','/workspace/training/exports/selected/inference.yml','--output','/workspace/full_receipt_20261117/prepared_v1'))
Run-Stage 'crop_val' ($base + @('/workspace/training/korie.py','evaluate','--manifest','/workspace/full_receipt_20261117/prepared_v1/manifest.jsonl','--data-root','/workspace/full_receipt_20261117/data/ocr','--baseline','/workspace/full_receipt_20261117/models/korean_PP-OCRv5_mobile_rec','--candidate','/workspace/training/exports/selected','--split','val','--output','/workspace/full_receipt_20261117/evaluation/pre_round2_val'))
$full = @('/workspace/full_receipt_20261117/full_receipt_v1.py','--images','/workspace/full_receipt_20261117/data/detection/val/images','--detector','/workspace/full_receipt_20261117/models/PP-OCRv5_mobile_det','--baseline','/workspace/full_receipt_20261117/models/korean_PP-OCRv5_mobile_rec','--candidate','/workspace/training/exports/selected')
Run-Stage 'full_smoke' ($base + $full + @('--limit','20','--output','/workspace/full_receipt_20261117/evaluation/full_val_smoke_v1'))
Run-Stage 'full_val' ($base + $full + @('--output','/workspace/full_receipt_20261117/evaluation/full_val_v1'))
Write-Output "Validation finished: $record"
