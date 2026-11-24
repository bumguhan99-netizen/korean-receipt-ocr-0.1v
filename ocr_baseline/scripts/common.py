import csv
import hashlib
import json
from pathlib import Path

MODELS = ('PP-OCRv5_mobile_rec', 'korean_PP-OCRv5_mobile_rec')
DEFAULT_DATA = Path(r'C:\Users\SSAFY\Desktop\mogu\mogumogu\dataset\korie')
MANIFEST_FIELDS = ['eval_split', 'source_split', 'receipt_id', 'field', 'image_path', 'gt_text', 'image_sha256']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))
