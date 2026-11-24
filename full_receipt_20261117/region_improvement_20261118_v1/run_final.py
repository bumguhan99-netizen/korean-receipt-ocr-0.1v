import subprocess,json,hashlib
from pathlib import Path
root=Path('/experiment'); base=['python','/experiment/region_experiment.py']
frozen=dict(resolution=1920,limit_type='max',thresh=.3,box_thresh=.45,unclip_ratio=1.2,merge=True,selection='lowest development CER among prespecified evaluated candidates; 48 source-hash groups; frozen before remaining 256 receipts',scope='Previously inspected diagnostic corpus, not untouched test',script_sha256=hashlib.sha256((root/'region_experiment.py').read_bytes()).hexdigest())
with (root/'frozen_region_policy.json').open('x') as f:json.dump(frozen,f,indent=2)
subprocess.run(base+['infer','--output','/experiment/final_1920','--subset','all'],check=True)
for name,opts in [('final_plain',[]),('final_merge',['--merge'])]:subprocess.run(base+['score','--input','/experiment/final_1920','--output','/experiment/'+name]+opts,check=True)
