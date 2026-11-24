param([Parameter(Mandatory=$true)][string]$ReviewJson, [string]$Root = '')
$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Join-Path $PSScriptRoot '../receipt_labeled' }
$Root = (Resolve-Path -LiteralPath $Root).Path
$payload = Get-Content -LiteralPath $ReviewJson -Raw -Encoding UTF8 | ConvertFrom-Json
if ($payload.schema_version -ne 1 -or -not $payload.edits) { throw 'Invalid review format' }
$labels = @{}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $Root 'labels') -Filter 'r*.json') { $labels[$file.BaseName] = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json }
# Validate the complete correction set before changing files.
foreach ($entry in $payload.edits.PSObject.Properties) {
    $edit = $entry.Value
    if (-not $labels.ContainsKey($edit.receipt_id)) { throw "Unknown receipt: $($entry.Name)" }
    $label = $labels[$edit.receipt_id]
    $line = @($label.lines | Where-Object id -eq $entry.Name)
    if ($line.Count -ne 1 -or $label.source_sha256 -ne $edit.source_sha256 -or $edit.text -isnot [string] -or $edit.status -notin @('verified','excluded','needs_review')) { throw "Invalid correction or source mismatch: $($entry.Name)" }
    if ($edit.status -eq 'verified' -and ([string]::IsNullOrWhiteSpace($edit.text) -or $edit.text.Contains("`t") -or $edit.text.Contains("`r") -or $edit.text.Contains("`n"))) { throw "Verified recognition label must be non-empty and single-line: $($entry.Name)" }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$backup = Join-Path $Root "backups/$stamp"
$null = New-Item -ItemType Directory -Path $backup -Force
Copy-Item -LiteralPath (Join-Path $Root 'labels') -Destination $backup -Recurse
foreach ($entry in $payload.edits.PSObject.Properties) {
    $edit = $entry.Value
    $line = $labels[$edit.receipt_id].lines | Where-Object id -eq $entry.Name
    if (-not $line.PSObject.Properties['original_ocr_text']) { $line | Add-Member NoteProperty original_ocr_text $line.text }
    $line.text = $edit.text
    $line.status = $edit.status
    $line | Add-Member -Force NoteProperty words_status 'original_ocr_unreviewed_do_not_use_as_gt'
    $line | Add-Member -Force NoteProperty reviewed_at (Get-Date -Format o)
    foreach ($key in @('review_method','review_sheet','review_note')) { if ($edit.PSObject.Properties[$key]) { $line | Add-Member -Force NoteProperty $key $edit.$key } }
}
$utf8 = New-Object Text.UTF8Encoding($false)
$training = New-Object 'System.Collections.Generic.List[string]'
$draftTraining = New-Object 'System.Collections.Generic.List[string]'
$rows = @()
# Archive labels that no longer belong to the verified training set.
$verifiedIds = @{}
foreach ($label in $labels.Values) { foreach ($line in $label.lines) { if ($line.status -eq 'verified') { $verifiedIds[$line.id] = $true } } }
$verifiedDirectory = Join-Path $Root 'verified_labels'
if (Test-Path -LiteralPath $verifiedDirectory) {
    foreach ($file in Get-ChildItem -LiteralPath $verifiedDirectory -Filter '*.txt' -File) {
        if (-not $verifiedIds.ContainsKey($file.BaseName)) {
            $archive = Join-Path $backup 'obsolete_verified_labels'
            $null = New-Item -ItemType Directory -Path $archive -Force
            Move-Item -LiteralPath $file.FullName -Destination (Join-Path $archive $file.Name)
        }
    }
}
foreach ($id in @($labels.Keys | Sort-Object)) {
    $label = $labels[$id]
    $pending = @($label.lines | Where-Object status -eq 'needs_review').Count
    # Region coverage cannot be certified by the line-only review UI.
    $verified = @($label.lines | Where-Object status -eq 'verified').Count
    $label.annotation_status = if ($pending -eq 0 -and @($label.lines).Count -gt 0) { 'detected_lines_reviewed_coverage_unchecked' } elseif ($verified -gt 0) { 'partially_reviewed' } else { 'auto_labeled_needs_review' }
    foreach ($line in $label.lines) {
        if ($line.status -eq 'verified') {
            $target = Join-Path $Root "verified_labels/$($line.id).txt"
            $null = New-Item -ItemType Directory -Path (Split-Path $target) -Force
            [IO.File]::WriteAllText($target,$line.text,$utf8)
            $training.Add("$($line.crop)`t$($line.text)")
        }
        if ($line.status -eq 'needs_review' -and -not [string]::IsNullOrWhiteSpace($line.text) -and -not $line.text.Contains("`t") -and -not $line.text.Contains("`n") -and -not $line.text.Contains("`r")) { $draftTraining.Add("$($line.crop)`t$($line.text)") }
        $rows += [pscustomobject][ordered]@{receipt_id=$id;line_id=$line.id;image=$line.crop;text=$line.text;status=$line.status;original_name=$label.original_name}
    }
    [IO.File]::WriteAllText((Join-Path $Root "labels/$id.json"),(ConvertTo-Json $label -Depth 40),$utf8)
    [IO.File]::WriteAllText((Join-Path $Root "text/$id.txt"),(($label.lines | ForEach-Object text) -join "`n"),$utf8)
}
$rows | Export-Csv -LiteralPath (Join-Path $Root 'lines.csv') -NoTypeInformation -Encoding UTF8
[IO.File]::WriteAllLines((Join-Path $Root 'recognition_verified.txt'),$training.ToArray(),$utf8)
[IO.File]::WriteAllLines((Join-Path $Root 'recognition_draft.txt'),$draftTraining.ToArray(),$utf8)
[IO.File]::WriteAllText((Join-Path $Root 'annotations.json'),(ConvertTo-Json -InputObject @($labels.Keys | Sort-Object | ForEach-Object {$labels[$_]}) -Depth 40),$utf8)
[IO.File]::WriteAllText((Join-Path $Root 'review_status.json'),(ConvertTo-Json @{verified_lines=$training.Count;total_lines=$rows.Count;pending_lines=@($rows | Where-Object status -eq 'needs_review').Count;excluded_lines=@($rows | Where-Object status -eq 'excluded').Count;full_receipt_region_coverage_checked=$false} -Depth 5),$utf8)
$index = @(Import-Csv -LiteralPath (Join-Path $Root 'index.csv') -Encoding UTF8)
foreach ($row in $index) { if ($labels.ContainsKey($row.id)) { $row.status = $labels[$row.id].annotation_status; $row.line_count = @($labels[$row.id].lines).Count } }
$index | Export-Csv -LiteralPath (Join-Path $Root 'index.csv') -Encoding UTF8 -NoTypeInformation
$reportPath = Join-Path $Root 'report.json'
if (Test-Path -LiteralPath $reportPath) {
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $report.lines = $rows.Count
    $report.reviewed_images = @($labels.Values | Where-Object annotation_status -eq 'detected_lines_reviewed_coverage_unchecked').Count
    $report | Add-Member -Force NoteProperty verified_recognition_lines $training.Count
    $report | Add-Member -Force NoteProperty full_receipt_region_coverage_checked $false
    [IO.File]::WriteAllText($reportPath,(ConvertTo-Json $report -Depth 40),$utf8)
}
& (Join-Path $PSScriptRoot 'build_receipt_review.ps1') -Root $Root
Write-Output "Verified recognition lines: $($training.Count). Crops from a receipt must stay in the same split."
