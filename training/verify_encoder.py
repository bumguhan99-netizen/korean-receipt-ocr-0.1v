"""Verify every eligible label with the actual pinned PaddleOCR multi-label encoder."""
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'PaddleOCR'))
from ppocr.data.imaug.label_ops import MultiLabelEncode

encoder = MultiLabelEncode(max_text_length=25,
                          character_dict_path=str(root/'PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt'),
                          use_space_char=True, gtc_encode='NRTRLabelEncode')
counts = {}
for split in ('train', 'val', 'test'):
    count = 0
    for line in (root/f'data/{split}_list.txt').read_text(encoding='utf-8').splitlines():
        path, text = line.split('\t', 1)
        result = encoder({'label': text, 'image': b''})
        if result is None or int(result['length']) != len(text):
            raise ValueError(f'Encoder rejected/changed label: {path}')
        count += 1
    counts[split] = count
print(json.dumps({'status': 'passed', 'encoded_counts': counts}))
