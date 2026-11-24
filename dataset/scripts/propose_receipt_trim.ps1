param()
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System; using System.Drawing; using System.Drawing.Imaging; using System.Runtime.InteropServices; using System.Collections.Generic;
public class ReceiptTrim {
 public static int[] Bounds(Bitmap source) {
  using(var b=new Bitmap(source.Width,source.Height,PixelFormat.Format24bppRgb)) {
   using(var g=Graphics.FromImage(b)) g.DrawImageUnscaled(source,0,0);
   var d=b.LockBits(new Rectangle(0,0,b.Width,b.Height),ImageLockMode.ReadOnly,PixelFormat.Format24bppRgb);
   byte[] p=new byte[d.Stride*b.Height]; Marshal.Copy(d.Scan0,p,0,p.Length); b.UnlockBits(d);
   int[] hist=new int[256]; for(int y=0;y<b.Height;y++) for(int x=0;x<b.Width;x++){int i=y*d.Stride+x*3;hist[(p[i]+p[i+1]+p[i+2])/3]++;}
   int total=b.Width*b.Height, sum=0, low=0, high=255; for(int i=0;i<256;i++){sum+=hist[i];if(sum<total*.1)low=i;if(sum<total*.9)high=i;}
   int threshold=Math.Min(180,(int)(low+(high-low)*.45));
   bool[] active=new bool[b.Height]; for(int y=0;y<b.Height;y++){int n=0;for(int x=0;x<b.Width;x++){int i=y*d.Stride+x*3;if((p[i]+p[i+1]+p[i+2])/3<threshold)n++;}active[y]=n>=Math.Max(2,b.Width*.008);}
   var bands=new List<int[]>(); int begin=-1, gap=0;
   for(int y=0;y<=b.Height;y++){bool a=y<b.Height&&active[y];if(a){if(begin<0)begin=y;gap=0;}else if(begin>=0){gap++;if(gap>=3||y==b.Height){bands.Add(new int[]{begin,y-gap+1});begin=-1;}}}
   if(bands.Count<2)return null; int[] best=null; foreach(var q in bands)if(best==null||q[1]-q[0]>best[1]-best[0])best=q;
   int size=best[1]-best[0];if(size<8||size<b.Height*.35||size>b.Height*.9)return null;
   foreach(var q in bands)if(q!=best&&q[1]-q[0]>size*.6)return null;
   return new int[]{Math.Max(0,best[0]-1),Math.Min(b.Height,best[1]+1)};
  }
 }
}
'@
$root=(Resolve-Path (Join-Path $PSScriptRoot '../receipt_labeled')).Path
$dir=Join-Path $root 'trim_candidates'
$null=New-Item -ItemType Directory -Force -Path $dir
$rows=New-Object 'System.Collections.Generic.List[object]'
foreach($f in Get-ChildItem (Join-Path $root 'labels') -Filter 'r*.json' | Sort-Object Name){
 $label=Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
 foreach($line in $label.lines){
  if($line.status -ne 'excluded' -or $line.crop_rotation_degrees -or $line.review_note -notmatch '줄|유입|바코드'){continue}
  $img=New-Object Drawing.Bitmap((Join-Path $root $line.crop))
  try{
   $bounds=[ReceiptTrim]::Bounds($img)
   if(-not $bounds){continue}
   $rect=New-Object Drawing.Rectangle(0,$bounds[0],$img.Width,($bounds[1]-$bounds[0]))
   $crop=$img.Clone($rect,$img.PixelFormat)
   try{$crop.Save((Join-Path $dir ($line.id+'.png')),[Drawing.Imaging.ImageFormat]::Png)}finally{$crop.Dispose()}
   $rows.Add([pscustomobject]@{id=$line.id;receipt_id=$label.id;source_sha256=$label.source_sha256;text=$line.text;crop=('trim_candidates/'+$line.id+'.png');original_bbox=$line.bbox;bbox=@($line.bbox[0],($line.bbox[1]+$bounds[0]),$line.bbox[2],($line.bbox[1]+$bounds[1]))})
  }finally{$img.Dispose()}
 }
}
[IO.File]::WriteAllText((Join-Path $dir 'index.json'),(ConvertTo-Json -InputObject $rows.ToArray() -Depth 10),(New-Object Text.UTF8Encoding($false)))
Write-Output "Proposed crops: $($rows.Count). Not verified or applied."
