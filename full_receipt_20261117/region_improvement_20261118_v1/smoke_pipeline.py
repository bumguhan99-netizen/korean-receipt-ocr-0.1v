import json,sys
from pathlib import Path
sys.path.insert(0,'/experiment')
from receipt_regions_v1 import RegionOCR,merge_fragments
def d(b,t):return dict(bbox=b,text=t,score=.99,polygon=[])
assert len(merge_fragments([d([0,0,130,20],'8801234567890'),d([150,0,200,20],'3,900')]))==1
assert len(merge_fragments([d([0,0,130,20],'8801234567890'),d([150,30,200,50],'3,900')]))==2
assert len(merge_fragments([d([0,0,20,20],'1'),d([150,0,180,20],'2')]))==2
assert len(merge_fragments([d([0,0,20,20],'부'),d([25,0,45,20],'가')]))==1
assert merge_fragments([])==[]
p=Path('/workspace/full_receipt_20261117'); model=RegionOCR(p/'models/PP-OCRv5_mobile_det',p/'round2/run_001/export')
result=model.predict('/workspace/dataset/receipt_labeled/aligned_images/r001.png')
expected=json.loads(Path('/experiment/final_1920/r001.json').read_text())['detections']
assert result['raw_detections']==expected
assert result['regions']==merge_fragments(expected)
with Path('/experiment/pipeline_verified.json').open('x') as f:json.dump(dict(status='passed',geometry_tests=5,full_image='r001',exact_raw_and_merged_reproduction=True),f,indent=2)
print('5 geometry tests passed; standalone pipeline matches evaluated predictions exactly')
