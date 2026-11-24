"""Record receipt-ID overlaps without treating IDs as verified transcription links."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parent
data = root/'data'
result = {'warning':'Receipt-ID overlap alone does not prove image identity or box-transcription correspondence.', 'splits':{}}
train = list((data/'ocr/train').glob('*.jpg'))
train_ids = {re.match(r'IMG\d+',p.stem)[0] for p in train}
train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in train}
for split in ('val','test'):
    originals = sorted((data/'detection'/split/'images').iterdir())
    result['splits'][split] = {'original_count':len(originals), 'original_ids_also_in_ocr_train':[p.stem for p in originals if p.stem in train_ids], 'original_byte_duplicates_of_train_crops':[p.name for p in originals if hashlib.sha256(p.read_bytes()).hexdigest() in train_hashes]}
with (root/'audit_overlap_v1.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({s:{k:len(v) if isinstance(v,list) else v for k,v in row.items()} for s,row in result['splits'].items()}))
