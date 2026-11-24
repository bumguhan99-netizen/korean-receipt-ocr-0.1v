"""Build a local receipt corpus without modifying source images."""
import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ocr', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'dataset'
    for folder in ['images', 'labels', 'splits']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    sources = sorted(p for base in [ROOT / 'receipt', ROOT / 'datasets--HumynLabs--Korean_Receipts_Dataset' / 'snapshots'] for p in base.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif'})
    records, rejected = [], []
    name_index_path = out / 'name_index.json'
    names = json.loads(name_index_path.read_text(encoding='utf-8')) if name_index_path.exists() else {}
    for path in sources:
        source = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sample_id = 'receipt_' + hashlib.sha256(source.encode()).hexdigest()[:16]
        if sample_id not in names:
            names[sample_id] = f'{max((int(n) for n in names.values()), default=0)+1:03d}'
        display_name = names[sample_id]
        label_path = out / 'labels' / (display_name + '.json')
        try:
            with Image.open(path) as image:
                image.load()
                original_size = list(image.size)
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((2400, 2400))
                gray = image.convert('L')
                small = gray.resize((9, 8))
                pixels = list(small.getdata())
                dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                brightness = ImageStat.Stat(gray).mean[0]
                image.save(out / 'images' / (display_name + '.jpg'), quality=95)
                width, height = image.size
        except Exception as error:
            rejected.append({'source': source, 'error': str(error)})
            continue
        flags = []
        if min(width, height) < 400: flags.append('low_resolution')
        if brightness < 60: flags.append('dark_image')
        record = dict(id=sample_id, display_name=display_name, image=f'images/{display_name}.jpg', label=f'labels/{display_name}.json', source=source, source_sha256=digest, width=width, height=height, original_size=original_size, dhash=f'{dhash:016x}', quality_flags=flags, source_license='CC-BY-4.0' if source.startswith('datasets--') else 'unspecified')
        records.append(record)
        if not label_path.exists():
            write_json(label_path, dict(id=sample_id, document_type='receipt', annotation_status='pending', ocr=None, fields={'merchant_name': None, 'transaction_date': None, 'total_amount': None, 'currency': None, 'items': []}, fields_status='unlabeled', privacy_review_status='pending', review_notes=''))
    # Connected components keep identical bytes, related filenames and visual duplicates together.
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def family(r):
        name = Path(r['source']).stem
        return re.sub(r'-\d+$', '', name) if re.match(r'^\d{8}_\d{6}', name) else name
    pairs = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            distance = (int(a['dhash'],16) ^ int(b['dhash'],16)).bit_count()
            same = a['source_sha256'] == b['source_sha256']
            similar = distance <= 6 and abs(a['width']/a['height'] - b['width']/b['height']) < .15
            if same or similar or family(a) == family(b):
                parents[find(i)] = find(j)
                pairs.append(dict(first=a['id'], second=b['id'], exact=same, dhash_distance=distance))
    # Conservative visual review groups from the contact sheet, plus matching OCR.
    reviewed_pairs = [
        ('receipt_cab1886057efd62d', 'receipt_2b4b07c283df340e'),
        ('receipt_70f96ba05fc7081e', 'receipt_f8485150431e7485'),
        ('receipt_272fdd06d104b327', 'receipt_ece64f2b18c36636'),
    ]
    lookup = {r['id']:i for i,r in enumerate(records)}
    for first, second in reviewed_pairs:
        if first in lookup and second in lookup:
            parents[find(lookup[first])] = find(lookup[second])
            pairs.append(dict(first=first, second=second, reason='visual_review_same_receipt_candidate'))
    groups = {}
    for i, record in enumerate(records): groups.setdefault(find(i), []).append(record)
    shuffled = list(groups.values())
    random.Random(42).shuffle(shuffled)
    counts = Counter()
    targets = {'train': len(records)*.8, 'validation': len(records)*.1, 'test':len(records)*.1}
    for group in sorted(shuffled, key=len, reverse=True):
        split = max(targets, key=lambda key: targets[key]-counts[key])
        for record in group:
            record['split'] = split
            record['group_id'] = min(r['id'] for r in group)
        counts[split] += len(group)
    if not args.skip_ocr:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT/'scripts/ocr_windows.ps1'), '-DatasetRoot', str(out)], check=True)
    for split in targets:
        selected = [r for r in records if r['split'] == split]
        (out/'splits'/f'{split}.txt').write_text(''.join(r['id']+'\n' for r in selected), encoding='utf-8')
        with (out/f'{split}.jsonl').open('w', encoding='utf-8') as handle:
            for record in selected:
                label = json.loads((out/record['label']).read_text(encoding='utf-8-sig'))
                handle.write(json.dumps({**record, 'annotation':label}, ensure_ascii=False)+'\n')
    write_json(out/'manifest.json', records)
    write_json(name_index_path, names)
    with (out/'filename_mapping.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','source','image','label','split','group_id','source_license'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    report = dict(source_count=len(sources), valid_count=len(records), rejected=rejected, splits=dict(counts), groups=len(groups), duplicate_or_related_pairs=pairs, quality_flag_counts=dict(Counter(f for r in records for f in r['quality_flags'])), seed=42, ocr_labels_reviewed=False, structured_fields_labeled=False)
    write_json(out/'quality_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
