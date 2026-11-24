"""Freeze model choice and full-receipt test policy using completed validation only."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'round2/run_001'
crop=json.loads((run/'validation/summary.json').read_text())
info=json.loads((run/'validation/run_info.json').read_text())
full=json.loads((root/'evaluation/full_val_round2_v1/summary.json').read_text())
if info['status']!='completed' or info['split']!='val' or info['n']!=1785 or full['status']!='completed' or full['n']!=154:
    raise ValueError('Full crop and receipt validation required')
first=root.parent/'training/exports/selected'
second=run/'export'
improved=crop['candidate']['cer_pct']<=crop['baseline']['cer_pct'] and crop['candidate']['exact_match_pct']>=crop['baseline']['exact_match_pct']
selection=dict(selected='round2' if improved else 'round1',export=str(second if improved else first),criterion='Keep round2 only if full validation CER and EM are both non-worse; epoch chosen by internal val accuracy',validation=crop)
with (root/'final_selection_v1.json').open('x') as stream:
    json.dump(selection,stream,indent=2)
def hashes(path):
    return {f:hashlib.sha256((path/f).read_bytes()).hexdigest() for f in ('inference.json','inference.pdiparams','inference.yml')}
policy=dict(validation_status='completed',models={key:hashes(path) for key,path in [('baseline',first),('candidate',second),('detector',root/'models/PP-OCRv5_mobile_det')]},runner_sha256=hashlib.sha256((root/'full_receipt_v2.py').read_bytes()).hexdigest(),selection=selection['selected'],normalization='NFC',scope='Qualitative full OCR; incomplete GT does not support full CER/precision')
policy['packages']=json.loads((root/'records/eval_packages_v1.json').read_text(encoding='utf-8-sig'))
with (root/'frozen_full_test_v1.json').open('x') as stream:
    json.dump(policy,stream,indent=2)
print(json.dumps(selection,indent=2))
