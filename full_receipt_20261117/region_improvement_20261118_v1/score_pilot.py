import subprocess,sys,json
from pathlib import Path
base=['python','/experiment/region_experiment.py','score']
for name,extra in [('baseline_all',['--baseline']),('baseline_dev',['--baseline','--subset','dev']),('pilot_plain',['--input','/experiment/pilot_1920']),('pilot_merge',['--input','/experiment/pilot_1920','--merge'])]:
 subprocess.run(base+['--output','/experiment/'+name]+extra,check=True)
# Unit checks: preserve distinct rows and distant columns, merge known barcode + price rule.
sys.path.insert(0,'/experiment'); from region_experiment import merge_fragments
def d(b,t):return dict(bbox=b,text=t,score=.99,polygon=[])
assert len(merge_fragments([d([0,0,130,20],'8801234567890'),d([150,0,200,20],'3,900')]))==1
assert len(merge_fragments([d([0,0,130,20],'8801234567890'),d([150,30,200,50],'3,900')]))==2
assert len(merge_fragments([d([0,0,20,20],'1'),d([150,0,180,20],'2')]))==2
assert len(merge_fragments([d([0,0,20,20],'부'),d([25,0,45,20],'가')]))==1
summary=json.loads(Path('/experiment/baseline_all/summary.json').read_text())
assert summary['n']==8343 and summary['missing_gt']==1110 and summary['char_edits']==18084
print('Baseline exact replay and geometry tests passed')
