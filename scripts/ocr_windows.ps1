param([string]$DatasetRoot)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($Operation, [Type]$ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $engine) { throw 'Install the Korean Windows OCR language pack.' }
foreach ($file in Get-ChildItem (Join-Path $DatasetRoot 'images') -Filter *.jpg) {
    $storage = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($file.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($storage.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try { $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult]) } finally { $bitmap.Dispose() }
        $lines = @($result.Lines | ForEach-Object {
            @{ text = $_.Text; words = @($_.Words | ForEach-Object {
                $r = $_.BoundingRect
                @{ text = $_.Text; bbox = @($r.X, $r.Y, ($r.X + $r.Width), ($r.Y + $r.Height)) }
            }) }
        })
        $labelPath = Join-Path $DatasetRoot ('labels/' + $file.BaseName + '.json')
        $label = Get-Content -LiteralPath $labelPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $label.ocr = @{ engine = 'Windows.Media.Ocr'; language = $engine.RecognizerLanguage.LanguageTag; text = $result.Text; lines = $lines; coordinate_format = 'xyxy_pixels'; confidence = $null }
        $label.annotation_status = 'auto_labeled_needs_review'
        $json = $label | ConvertTo-Json -Depth 30
        [IO.File]::WriteAllText($labelPath, $json, (New-Object Text.UTF8Encoding($false)))
        Write-Output $file.BaseName
    } finally { $stream.Dispose() }
}
