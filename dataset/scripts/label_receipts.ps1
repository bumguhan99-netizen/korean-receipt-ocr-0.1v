param([string]$SourceRoot = (Join-Path $PSScriptRoot '../receipt_imgs'), [string]$OutputRoot = (Join-Path $PSScriptRoot '../receipt_labeled'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($Operation, [Type]$ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}
function SaveJson($Path, $Value) { [IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 40), (New-Object Text.UTF8Encoding($false))) }
$language = New-Object Windows.Globalization.Language('ko-KR')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if (-not $engine) { throw 'Korean Windows OCR language pack is required.' }
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
foreach ($dir in @('', 'images', 'labels', 'text', 'crops', 'draft_labels', 'previews')) { $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutputRoot $dir) }
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$files = @(Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Where-Object { $_.Name -notlike '._*' -and $_.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png') } | Sort-Object FullName)
$oldMapping = @{}
$mappingPath = Join-Path $OutputRoot 'index.csv'
if (Test-Path -LiteralPath $mappingPath) { foreach ($row in Import-Csv -LiteralPath $mappingPath -Encoding UTF8) { $oldMapping[$row.original_name] = $row.id } }
$nextId = 1
foreach ($id in $oldMapping.Values) { $n = [int]$id.Substring(1); if ($n -ge $nextId) { $nextId = $n + 1 } }
$rows = New-Object 'System.Collections.Generic.List[object]'
$errors = New-Object 'System.Collections.Generic.List[object]'
$allLabels = New-Object 'System.Collections.Generic.List[object]'
foreach ($file in $files) {
    $relative = $file.FullName.Substring($SourceRoot.Length + 1).Replace('\','/')
    if ($oldMapping.ContainsKey($relative)) { $id = $oldMapping[$relative] } else { $id = 'r{0:D3}' -f $nextId; $nextId++ }
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $labelPath = Join-Path $OutputRoot "labels/$id.json"
    try {
        if (Test-Path -LiteralPath $labelPath) {
            $label = Get-Content -LiteralPath $labelPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($label.source_sha256 -ne $hash) { throw 'Source hash changed; refusing to overwrite labels.' }
        } else {
            $src = [Drawing.Image]::FromFile($file.FullName)
            try {
                if ($src.PropertyIdList -contains 274) {
                    $orientation = [BitConverter]::ToUInt16($src.GetPropertyItem(274).Value,0)
                    $rotation = @{2=4;3=2;4=6;5=5;6=1;7=7;8=3}
                    if ($rotation.ContainsKey([int]$orientation)) { $src.RotateFlip([Drawing.RotateFlipType]$rotation[[int]$orientation]) }
                }
                # Save lossless, display-oriented annotation copy; never change the source.
                $bitmap = New-Object Drawing.Bitmap($src.Width, $src.Height)
                $g = [Drawing.Graphics]::FromImage($bitmap)
                try { $g.DrawImage($src, 0, 0, $src.Width, $src.Height) } finally { $g.Dispose() }
                $bitmap.Save((Join-Path $OutputRoot "images/$id.png"), [Drawing.Imaging.ImageFormat]::Png)
            } finally { $src.Dispose() }
            try {
                $scale = [Math]::Min(1.0, ([Windows.Media.Ocr.OcrEngine]::MaxImageDimension / [double][Math]::Max($bitmap.Width,$bitmap.Height)))
                $ocrWidth = [int][Math]::Floor($bitmap.Width * $scale)
                $ocrHeight = [int][Math]::Floor($bitmap.Height * $scale)
                $ocrBitmap = New-Object Drawing.Bitmap($bitmap, $ocrWidth, $ocrHeight)
                try { $ocrBitmap.Save((Join-Path $OutputRoot "previews/$id.png"), [Drawing.Imaging.ImageFormat]::Png) } finally { $ocrBitmap.Dispose() }
                $storage = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync((Join-Path $OutputRoot "previews/$id.png"))) ([Windows.Storage.StorageFile])
                $stream = Await ($storage.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
                try {
                    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
                    $softwareBitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
                    try { $result = Await ($engine.RecognizeAsync($softwareBitmap)) ([Windows.Media.Ocr.OcrResult]) } finally { $softwareBitmap.Dispose() }
                } finally { $stream.Dispose() }
                $lines = @()
                $lineNumber = 0
                foreach ($line in $result.Lines) {
                    $lineNumber++
                    $lineId = '{0}_{1:D3}' -f $id,$lineNumber
                    $words = @($line.Words | ForEach-Object {
                        $r = $_.BoundingRect
                        [ordered]@{text=$_.Text; bbox=@([Math]::Floor($r.X*$bitmap.Width/$ocrWidth),[Math]::Floor($r.Y*$bitmap.Height/$ocrHeight),[Math]::Ceiling(($r.X+$r.Width)*$bitmap.Width/$ocrWidth),[Math]::Ceiling(($r.Y+$r.Height)*$bitmap.Height/$ocrHeight))}
                    })
                    $x1 = [int](($words | ForEach-Object {$_.bbox[0]} | Measure-Object -Minimum).Minimum)
                    $y1 = [int](($words | ForEach-Object {$_.bbox[1]} | Measure-Object -Minimum).Minimum)
                    $x2 = [int](($words | ForEach-Object {$_.bbox[2]} | Measure-Object -Maximum).Maximum)
                    $y2 = [int](($words | ForEach-Object {$_.bbox[3]} | Measure-Object -Maximum).Maximum)
                    $x1 = [Math]::Max(0,$x1-2); $y1 = [Math]::Max(0,$y1-2)
                    $x2 = [Math]::Min($bitmap.Width,$x2+2); $y2 = [Math]::Min($bitmap.Height,$y2+2)
                    if ($x2 -le $x1 -or $y2 -le $y1) { throw "Invalid box: $lineId" }
                    $crop = $bitmap.Clone((New-Object Drawing.Rectangle($x1,$y1,($x2-$x1),($y2-$y1))),[Drawing.Imaging.PixelFormat]::Format24bppRgb)
                    try { $crop.Save((Join-Path $OutputRoot "crops/$lineId.png"),[Drawing.Imaging.ImageFormat]::Png) } finally { $crop.Dispose() }
                    [IO.File]::WriteAllText((Join-Path $OutputRoot "draft_labels/$lineId.txt"),$line.Text,(New-Object Text.UTF8Encoding($false)))
                    $lines += [ordered]@{id=$lineId; text=$line.Text; bbox=@($x1,$y1,$x2,$y2); words=$words; crop="crops/$lineId.png"; status='needs_review'}
                }
                $label = [ordered]@{schema_version=1; id=$id; original_name=$relative; source_sha256=$hash; image="images/$id.png"; width=$bitmap.Width; height=$bitmap.Height; annotation_status='auto_labeled_needs_review'; engine='Windows.Media.Ocr'; language=$engine.RecognizerLanguage.LanguageTag; coordinate_format='xyxy_pixels'; text_angle=$result.TextAngle; lines=@($lines); fields=$null; training_ready=$false; notes='Automatic transcription and regions require visual review. No receipt fields inferred.'}
                SaveJson $labelPath $label
                [IO.File]::WriteAllText((Join-Path $OutputRoot "text/$id.txt"),(($lines | ForEach-Object {$_.text}) -join "`n"),(New-Object Text.UTF8Encoding($false)))
            } finally { $bitmap.Dispose() }
        }
        $allLabels.Add($label)
        $rows.Add([pscustomobject][ordered]@{id=$id;original_name=$relative;image=$label.image;label="labels/$id.json";text="text/$id.txt";source_sha256=$hash;line_count=@($label.lines).Count;status=$label.annotation_status})
        $rows | Export-Csv -LiteralPath $mappingPath -NoTypeInformation -Encoding UTF8
        Write-Output "$id $relative : $(@($label.lines).Count) lines"
    } catch {
        $errors.Add([ordered]@{id=$id;source=$relative;error=$_.Exception.ToString()})
        Write-Output "ERROR $id $relative : $_"
        # Preserve IDs even if one source fails.
        $rows.Add([pscustomobject][ordered]@{id=$id;original_name=$relative;image='';label='';text='';source_sha256=$hash;line_count=0;status='failed'})
        $rows | Export-Csv -LiteralPath $mappingPath -NoTypeInformation -Encoding UTF8
    }
}
SaveJson (Join-Path $OutputRoot 'annotations.json') @($allLabels.ToArray())
$duplicates = @($rows | Group-Object source_sha256 | Where-Object Count -gt 1 | ForEach-Object { [ordered]@{sha256=$_.Name;ids=@($_.Group.id)} })
SaveJson (Join-Path $OutputRoot 'report.json') ([ordered]@{source_images=$files.Count;successful=$allLabels.Count;failed=$errors.Count;errors=@($errors.ToArray());ignored_appledouble=@(Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Where-Object Name -Like '._*').Count;lines=($rows | Measure-Object line_count -Sum).Sum;duplicates=$duplicates;engine='Windows.Media.Ocr';language=$engine.RecognizerLanguage.LanguageTag;reviewed_images=0;training_ready_images=0;source_modified=$false})
if ($errors.Count) { throw "$($errors.Count) images failed; see report.json" }
& (Join-Path $PSScriptRoot 'repair_receipt_crops.ps1') -Root $OutputRoot
$currentLabels=@(Get-ChildItem (Join-Path $OutputRoot 'labels') -Filter 'r*.json' | Sort-Object Name | ForEach-Object { Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json })
SaveJson (Join-Path $OutputRoot 'annotations.json') $currentLabels
