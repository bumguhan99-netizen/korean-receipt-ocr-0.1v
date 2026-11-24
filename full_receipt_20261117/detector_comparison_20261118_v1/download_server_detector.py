import urllib.request,json,hashlib,time
from pathlib import Path
root=Path(__file__).parent/'server_detector_download_v2'; root.mkdir(exist_ok=False)
repo='PaddlePaddle/PP-OCRv5_server_det'
with urllib.request.urlopen('https://huggingface.co/api/models/'+repo,timeout=60) as f: metadata=json.load(f)
revision=metadata['sha']; records=[]
for name in ('inference.json','inference.yml','inference.pdiparams'):
 url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
 with urllib.request.urlopen(url,timeout=120) as source,(root/name).open('xb') as dest:
  while data:=source.read(1024*1024):dest.write(data)
 p=root/name; records.append(dict(name=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url=url)); print(name,p.stat().st_size,flush=True)
(root/'provenance.json').write_text(json.dumps(dict(repo=repo,revision=revision,files=records),indent=2),encoding='utf-8')
