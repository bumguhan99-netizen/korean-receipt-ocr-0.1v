param(
    [Parameter(Mandatory=$true)][string]$ImageRelativePath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$projectRoot = 'D:\ssafy\ocr\mogumogu'
$imageFull = [IO.Path]::GetFullPath((Join-Path $projectRoot $ImageRelativePath))
if (-not $imageFull.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Image must be inside the project.' }
if (-not (Test-Path -LiteralPath $imageFull -PathType Leaf)) { throw 'Image does not exist.' }
$outFull = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outFull) { throw 'Use a new output directory; existing results are preserved.' }
New-Item -ItemType Directory -Path $outFull | Out-Null
$imageContainer = '/workspace/' + [IO.Path]::GetRelativePath($projectRoot,$imageFull).Replace('\','/')
$dockerPath = 'C:\Users\hbeom\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
& $dockerPath run --rm --gpus all --network none --shm-size 2g `
    -v "${projectRoot}:/workspace:ro" -v "${PSScriptRoot}:/runner:ro" -v "${outFull}:/result" `
    --entrypoint python mogumogu-korie-training:20261117 /runner/receipt_regions_v1.py `
    --image $imageContainer `
    --detector /workspace/full_receipt_20261117/models/PP-OCRv5_mobile_det `
    --model /workspace/full_receipt_20261117/round2/run_001/export `
    --output /result/prediction.json
if ($LASTEXITCODE -ne 0) { throw 'OCR execution failed; retain the output directory for diagnosis.' }
Write-Output "Saved: $outFull\prediction.json"
