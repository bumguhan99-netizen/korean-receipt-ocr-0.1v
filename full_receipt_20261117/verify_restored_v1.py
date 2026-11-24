"""Export restored training weights and require identical inference parameters."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

root = Path(__file__).resolve().parent
repo = root/'PaddleOCR'
out = root/'restored_identity_v1'
out.mkdir(exist_ok=False)
weights = root/'restored_round1/best_accuracy.pdparams'
cfg = yaml.safe_load((root.parent/'training/configs/official.yml').read_text())
cfg['Global'].update(pretrained_model=str(weights), character_dict_path=str(repo/'ppocr/utils/dict/ppocrv5_korean_dict.txt'), save_inference_dir=str(out/'export'), checkpoints=None)
(out/'config.yml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
with (out/'export.log').open('x') as log:
    subprocess.run([sys.executable,str(repo/'tools/export_model.py'),'-c',str(out/'config.yml')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
expected = root.parent/'training/exports/selected/inference.pdiparams'
actual = out/'export/inference.pdiparams'
receipt = dict(status='passed' if digest(actual)==digest(expected) else 'failed', training_sha256=digest(weights), restored_export_sha256=digest(actual), selected_export_sha256=digest(expected))
(out/'identity.json').write_text(json.dumps(receipt,indent=2))
print(receipt)
if receipt['status'] != 'passed':
    raise RuntimeError('Export parameters differ; investigate before training')
