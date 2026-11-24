param([string]$OnlyId='', [string]$Root='')
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
if (-not $Root) { $Root=Join-Path $PSScriptRoot '../receipt_labeled' }
$root=(Resolve-Path -LiteralPath $Root).Path
$null=New-Item -ItemType Directory -Force -Path (Join-Path $root 'aligned_images')
$backup=Join-Path $root ('backups/before_coordinate_repair_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
$null=New-Item -ItemType Directory -Force -Path $backup
Copy-Item -LiteralPath (Join-Path $root 'labels') -Destination $backup -Recurse
$count=0
foreach($file in Get-ChildItem (Join-Path $root 'labels') -Filter 'r*.json' | Sort-Object Name){
    if($OnlyId -and $file.BaseName -ne $OnlyId){continue}
    $l=Get-Content $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if($l.PSObject.Properties['crop_coordinate_repair']){continue}
    $angle=[double]$l.text_angle
    $src=[Drawing.Bitmap]::FromFile((Join-Path $root $l.image))
    $bitmap=New-Object Drawing.Bitmap($src.Width,$src.Height)
    try{
        $g=[Drawing.Graphics]::FromImage($bitmap)
        try{
            $g.Clear([Drawing.Color]::White)
            $g.InterpolationMode=[Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $g.TranslateTransform([single]($src.Width/2),[single]($src.Height/2))
            $g.RotateTransform([single](-$angle))
            $g.TranslateTransform([single](-$src.Width/2),[single](-$src.Height/2))
            $g.DrawImage($src,0,0,$src.Width,$src.Height)
        }finally{$g.Dispose()}
        $bitmap.Save((Join-Path $root "aligned_images/$($l.id).png"),[Drawing.Imaging.ImageFormat]::Png)
        foreach($line in $l.lines){
            if($line.method -eq 'assistant_manual_region'){continue}
            $b=$line.bbox
            $x1=[Math]::Max(0,[int]$b[0]);$y1=[Math]::Max(0,[int]$b[1]);$x2=[Math]::Min($bitmap.Width,[int]$b[2]);$y2=[Math]::Min($bitmap.Height,[int]$b[3])
            $crop=$bitmap.Clone((New-Object Drawing.Rectangle($x1,$y1,($x2-$x1),($y2-$y1))),[Drawing.Imaging.PixelFormat]::Format24bppRgb)
            try{$crop.Save((Join-Path $root $line.crop),[Drawing.Imaging.ImageFormat]::Png)}finally{$crop.Dispose()}
            $line | Add-Member -Force NoteProperty coordinate_image "aligned_images/$($l.id).png"
        }
        $l | Add-Member -Force NoteProperty crop_coordinate_repair ([pscustomobject]@{rotation_degrees=(-$angle);center=@(($src.Width/2),($src.Height/2));coordinate_image="aligned_images/$($l.id).png";source_image_unchanged=$true;method='rotate_image_counterclockwise_by_OcrResult.TextAngle';applied_at=(Get-Date -Format o)})
        [IO.File]::WriteAllText($file.FullName,(ConvertTo-Json $l -Depth 40),(New-Object Text.UTF8Encoding($false)))
        $count++
    }finally{$bitmap.Dispose();$src.Dispose()}
}
Write-Output "Coordinate repair: $count images"
