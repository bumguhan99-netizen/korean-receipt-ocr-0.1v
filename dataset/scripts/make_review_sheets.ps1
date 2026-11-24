param([int]$Start=1,[int]$Count=1,[int]$PageSize=60,[string]$IndexPath='', [string]$OutDir='review_sheets')
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=(Resolve-Path (Join-Path $PSScriptRoot '../receipt_labeled')).Path
$out=Join-Path $root $OutDir
$null=New-Item -ItemType Directory -Force -Path $out
$all=@(Get-ChildItem (Join-Path $root 'labels') -Filter 'r*.json' | Sort-Object Name | ForEach-Object {
    $label=Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach($line in $label.lines){if($line.status -eq 'needs_review'){[pscustomobject]@{id=$line.id;text=$line.text;crop=$line.crop;receipt_id=$label.id;source_sha256=$label.source_sha256}}}
})
if (-not $IndexPath) { $IndexPath=Join-Path $out 'index.json' }
if(Test-Path $indexPath){$all=Get-Content $indexPath -Raw -Encoding UTF8 | ConvertFrom-Json}else{[IO.File]::WriteAllText($indexPath,(ConvertTo-Json -InputObject $all -Depth 10),(New-Object Text.UTF8Encoding($false)))}
$font=New-Object Drawing.Font('Malgun Gothic',11)
try{
foreach($page in $Start..($Start+$Count-1)){
    $items=@($all | Select-Object -Skip (($page-1)*$PageSize) -First $PageSize)
    if(-not $items.Count){break}
    $sheet=New-Object Drawing.Bitmap(2000,([int][Math]::Ceiling($items.Count/2)*112))
    $g=[Drawing.Graphics]::FromImage($sheet);$g.Clear([Drawing.Color]::White)
    try{
        $i=0
        foreach($line in $items){
            $x=($i%2)*1000;$y=[int][Math]::Floor($i/2)*112
            $caption=$line.id+'  '+$line.text
            if($caption.Length -gt 100){$caption=$caption.Substring(0,100)}
            $g.DrawString($caption,$font,[Drawing.Brushes]::Black,[single]($x+4),[single]($y+2))
            $img=[Drawing.Image]::FromFile((Join-Path $root $line.crop))
            try{$scale=[Math]::Min(985.0/$img.Width,72.0/$img.Height);$g.DrawImage($img,[int]($x+4),[int]($y+37),[int]($img.Width*$scale),[int]($img.Height*$scale))}finally{$img.Dispose()}
            $g.DrawLine([Drawing.Pens]::LightGray,$x,$y+111,$x+999,$y+111)
            $i++
        }
        $sheet.Save((Join-Path $out ('page_{0:D3}.png' -f $page)),[Drawing.Imaging.ImageFormat]::Png)
    }finally{$g.Dispose();$sheet.Dispose()}
    Write-Output ('page_{0:D3}: {1} - {2}' -f $page,$items[0].id,$items[-1].id)
}
}finally{$font.Dispose()}
