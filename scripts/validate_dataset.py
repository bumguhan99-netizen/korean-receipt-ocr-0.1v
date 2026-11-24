"""Check image-label alignment, OCR coordinates, split isolation and exports."""
import json
from pathlib import Path
from PIL import Image

root=Path(__file__).resolve().parents[1]/'dataset'
records=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
ids=set();groups={};hashes={};words=0;empty=[]
for r in records:
    assert r['id'] not in ids, 'Duplicate id'
    ids.add(r['id'])
    for mapping,key in [(groups,'group_id'),(hashes,'source_sha256')]:
        assert mapping.setdefault(r[key],r['split']) == r['split'], 'Split leakage'
    with Image.open(root/r['image']) as im:
        im.verify()
    label=json.loads((root/r['label']).read_text(encoding='utf-8-sig'))
    assert label['id']==r['id']
    assert label['ocr'] is not None, f'Missing OCR: {r["id"]}'
    if not label['ocr']['text'].strip(): empty.append(r['id'])
    for line in label['ocr']['lines']:
        for word in line['words']:
            x1,y1,x2,y2=word['bbox']
            assert 0<=x1<=x2<=r['width']+1 and 0<=y1<=y2<=r['height']+1, (r['id'],word)
            words+=1
for split in ['train','validation','test']:
    exported=[json.loads(line) for line in (root/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['id'] for r in exported} == {r['id'] for r in records if r['split']==split}
    for r in exported:
        assert r['annotation']==json.loads((root/r['label']).read_text(encoding='utf-8-sig')), 'Stale export'
print(json.dumps({'images':len(records),'ocr_words':words,'empty_ocr':empty,'validation':'passed'},indent=2))
