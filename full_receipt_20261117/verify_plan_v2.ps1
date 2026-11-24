# Read-only readiness audit. Windows PowerShell 5.1+, no Python or downloads.
[CmdletBinding()]
param(
    [string]$DataRoot,
    [string]$Archive,
    [string]$SourceUrl,
    [string]$SourceRevision,
    [string]$DetectorDir,
    [string]$BaselineDir,
    [string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
if (-not $PSScriptRoot) { $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot 'verification' }
$project = Split-Path $PSScriptRoot -Parent
$checks = New-Object 'System.Collections.Generic.List[object]'
function Check($Name, $Status, $Detail) {
    $checks.Add([pscustomobject]@{name=$Name; status=$Status; detail=$Detail})
}
function Model($Name, $Directory) {
    if (-not $Directory -or -not (Test-Path -LiteralPath $Directory -PathType Container)) {
        Check $Name 'NEEDS_CONFIRMATION' 'Local model directory missing or unspecified'; return
    }
    $files = @()
    foreach ($filename in @('inference.json','inference.pdiparams','inference.yml')) {
        $path = Join-Path $Directory $filename
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $files += @{name=$filename; bytes=(Get-Item -LiteralPath $path).Length; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash}
        } else { Check "$Name/$filename" 'NEEDS_CONFIRMATION' 'Missing expected export file' }
    }
    Check $Name 'INVENTORY_ONLY' $files
}
# Never put audit outputs inside a supplied original data directory.
$output = [IO.Path]::GetFullPath($OutputRoot)
if ($DataRoot) {
    $data = (Resolve-Path -LiteralPath $DataRoot).Path
    if ($output.TrimEnd('\') -eq $data.TrimEnd('\') -or $output.StartsWith($data.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must be outside DataRoot'
    }
}
$run = Join-Path $output ((Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+[guid]::NewGuid().ToString('N').Substring(0,8))
$null = New-Item -ItemType Directory -Path $run
Check 'host' 'INFO' @{os=[Environment]::OSVersion.VersionString; powershell=$PSVersionTable.PSVersion.ToString(); project=$project; cpu=$env:PROCESSOR_IDENTIFIER}
Check 'storage' 'INFO' @(Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free)
foreach ($command in @('nvidia-smi','nvcc','wsl','docker','py','python','python3.12')) {
    $found = Get-Command $command -ErrorAction SilentlyContinue
    Check "command/$command" $(if ($found) {'FOUND_NOT_EXECUTED'} else {'MISSING'}) @($found | Select-Object Name,Source)
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
    Check 'gpu' $(if ($LASTEXITCODE -eq 0) {'INFO'} else {'FAIL'}) ($gpu -join "`n")
}
Check 'gpu_compatibility' 'NEEDS_CONFIRMATION' 'Driver/UMD is not local CUDA Toolkit. Paddle import, GPU kernels and OCR inference not tested.'
Model 'candidate' (Join-Path $project 'training/exports/selected')
if (-not $BaselineDir) {
    $initial = Join-Path $project 'training/exports/initial'
    if (Test-Path -LiteralPath $initial -PathType Container) { $BaselineDir = $initial }
}
Model 'baseline' $BaselineDir
Model 'detector' $DetectorDir
foreach ($relative in @(
    'training/korie.py','training/test_korie.py','training/gpu_check.py',
    'training/requirements.txt','training/requirements.lock.txt','training/RESULTS.md',
    'training/.venv/bin/python','training/PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt',
    'training/data/manifest.jsonl','training/data/validation_report.json',
    'training/data/issues.csv','training/evaluation/selected_val/run_info.json',
    'training/evaluation/final_test/summary.json','training/records/packages.txt',
    'training/records/paddleocr_commit.txt','training/exports/initial/inference.pdiparams',
    'training/outputs/official/best_accuracy.pdparams','training/outputs/official/best_accuracy.pdopt'
)) {
    $path = Join-Path $project $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Check "training_asset/$relative" 'PRESENT' @{bytes=(Get-Item -LiteralPath $path).Length; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash}
    } else { Check "training_asset/$relative" 'NEEDS_CONFIRMATION' 'Not present on this PC; historical results are not a reproduced run.' }
}
Check 'existing_crop_pipeline' 'INFO' @{
    implementation='training/korie.py: prepare, verify, evaluate, freeze'
    metrics='NFC CER, EM, numeric-symbol EM, explicit amount-field EM; improved/regressed CSV'
    reuse='Use existing crop evaluator after original manifest, source crops and baseline export are restored.'
    limitation='Recognition only. Does not run receipt detection or box matching.'
}
if ($Archive) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archivePath = (Resolve-Path -LiteralPath $Archive).Path
    Check 'archive_provenance' 'INFO' @{path=$archivePath; url=$SourceUrl; revision=$SourceRevision; bytes=(Get-Item -LiteralPath $archivePath).Length; sha256=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash}
    if (-not $SourceUrl -or -not $SourceRevision) { Check 'archive_source' 'NEEDS_CONFIRMATION' 'Provide official source URL and revision/date; hash alone does not authenticate origin.' }
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $entries = @($zip.Entries | ForEach-Object {
            $unsafe = $_.FullName -match '(^[/\\]|^[A-Za-z]:|(^|[/\\])\.\.([/\\]|$))'
            if ($unsafe) { Check 'zip_path' 'FAIL' $_.FullName }
            [pscustomobject]@{path=$_.FullName; bytes=$_.Length; compressed_bytes=$_.CompressedLength}
        })
        $entries | Export-Csv -LiteralPath (Join-Path $run 'zip_inventory.csv') -NoTypeInformation -Encoding UTF8
        Check 'zip' 'INVENTORY_ONLY' @{entries=$entries.Count; uncompressed_bytes=($entries | Measure-Object bytes -Sum).Sum; note='Not extracted. Payload CRC and annotation semantics not verified.'}
    } finally { $zip.Dispose() }
}
if ($DataRoot) {
    Add-Type -AssemblyName System.Drawing
    $inventory = @()
    foreach ($file in Get-ChildItem -LiteralPath $data -File -Recurse) {
        $relative = $file.FullName.Substring($data.Length).TrimStart('\','/')
        $row = [ordered]@{path=$relative; bytes=$file.Length; type=$file.Extension; width=$null; height=$null; status='inventory'; error=$null}
        try {
            if ($file.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png','.bmp','.tif','.tiff')) {
                $image = [Drawing.Image]::FromFile($file.FullName)
                try { $row.width=$image.Width; $row.height=$image.Height; $row.status='image_loaded' } finally { $image.Dispose() }
            } elseif ($file.Extension -eq '.json') {
                $null = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
                $row.status='json_syntax_only'
            } elseif ($file.Extension -eq '.jsonl') {
                $n=0
                foreach ($line in [IO.File]::ReadLines($file.FullName)) {
                    $n++; if (-not [string]::IsNullOrWhiteSpace($line)) { $null = $line | ConvertFrom-Json }
                }
                $row.status="jsonl_syntax_only:$n"
            }
        } catch { $row.status='failed'; $row.error=$_.Exception.Message; Check 'file_read' 'FAIL' @{path=$relative; error=$row.error} }
        $inventory += [pscustomobject]$row
    }
    $inventory | Export-Csv -LiteralPath (Join-Path $run 'data_inventory.csv') -NoTypeInformation -Encoding UTF8
    Check 'data_counts' 'INFO' @($inventory | Group-Object type | Select-Object Name,Count)
    if ($inventory.Count -eq 0) { Check 'data_empty' 'FAIL' $data }
} else { Check 'data' 'NEEDS_CONFIRMATION' 'Pass the actual extracted official KORIE root; project dataset is not assumed to be KORIE.' }
foreach ($item in @('official_split_mapping','box_coordinate_schema','receipt_crop_id_link','text_gt_coverage','dictionary_and_length','training_receipt_overlap','fixed_detector_predictions','validation_frozen_scoring')) {
    Check $item 'NEEDS_CONFIRMATION' 'Requires verified official annotation schema; inventory success does not mean OCR evaluation is ready.'
}
$report = @{created=(Get-Date).ToString('o'); scope='read-only readiness and inventory, not accuracy evaluation'; checks=@($checks.ToArray()); ready_for_e2e_evaluation=$false}
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $run 'report.json') -Encoding UTF8
$checks | Select-Object name,status | Format-Table -AutoSize
Write-Output "Report: $run"
if (@($checks | Where-Object status -eq 'FAIL').Count) { exit 1 }
# 2 explicitly means manual/schema/runtime checks remain, not a successful OCR run.
exit 2
