param([Parameter(Mandatory=$true)][string]$Id, [ValidateSet(0,90,180,270)][int]$Degrees=180)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=(Resolve-Path (Join-Path $PSScriptRoot '../receipt_labeled')).Path
$file=Join-Path $root "labels/$Id.json"
$label=Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
$backup=Join-Path $root ('backups/orientation_'+$Id+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
$null=New-Item -ItemType Directory -Path $backup
Copy-Item -LiteralPath $file -Destination $backup
foreach($line in $label.lines){
 $previous=[int]$line.crop_rotation_degrees
 if($previous -eq $Degrees){continue}
 $delta=($Degrees-$previous+360)%360
 $p=Join-Path $root $line.crop
 Copy-Item -LiteralPath $p -Destination $backup
 $src=[Drawing.Bitmap]::FromFile($p)
 $bmp=New-Object Drawing.Bitmap($src)
 $src.Dispose()
 try{$bmp.RotateFlip([Drawing.RotateFlipType]("Rotate${delta}FlipNone"));$bmp.Save($p,[Drawing.Imaging.ImageFormat]::Png)}finally{$bmp.Dispose()}
 $line | Add-Member -Force NoteProperty crop_rotation_degrees $Degrees
}
[IO.File]::WriteAllText($file,(ConvertTo-Json $label -Depth 40),(New-Object Text.UTF8Encoding($false)))
