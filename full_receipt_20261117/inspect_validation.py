"""Audit downloaded validation only; never infer unprovided bbox transcriptions."""
import collections
import csv
import json
from pathlib import Path
import re
import zipfile

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent


def extract(name):
    destination = ROOT / 'extracted' / name
    marker = destination / '.extraction_complete'
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / 'archives' / f'{name}.zip') as archive:
        for entry in archive.infolist():
            normalized = entry.filename.replace('\\', '/')
            target = (destination / normalized).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe ZIP member: {entry.filename}')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(entry)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing extracted file differs: {target}')
            else:
                target.write_bytes(content)
    marker.write_text('ZIP CRC checked at acquisition; original archive preserved\n', encoding='utf-8')
    return destination


def main():
    output = ROOT / 'audit_validation'
    output.mkdir(exist_ok=False)
    detection = extract('detection_val') / 'val'
    ocr = extract('ocr_val') / 'val'
    classes = collections.Counter()
    issues, box_rows, crop_rows = [], [], []
    originals = {}
    for path in sorted((detection / 'images').iterdir()):
        if not path.is_file():
            continue
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        originals[path.stem] = path
        label = detection / 'labels' / (path.stem + '.txt')
        for index, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines()):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Non-YOLO label at {label}:{index+1}')
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if not all(0 <= x <= 1 for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                issues.append(dict(image=path.name, box=index, reason='invalid_normalized_coordinate'))
            bounds = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
            if bounds[0] < -1 or bounds[1] < -1 or bounds[2] > width+1 or bounds[3] > height+1:
                issues.append(dict(image=path.name, box=index, reason='box_outside_image'))
            box_rows.append(dict(receipt_id=path.stem, split='val', image=str(path.relative_to(ROOT)),
                                 box_id=index, class_id=cls, bbox_xyxy=bounds, text=None,
                                 text_status='not_provided_in_detection_annotation'))
            classes[cls] += 1
    model = ROOT.parent / 'training/exports/selected/inference.yml'
    chars = set(yaml.safe_load(model.read_text(encoding='utf-8'))['PostProcess']['character_dict'])
    chars.add(' ')
    for path in sorted(ocr.glob('*.jpg')):
        match = re.fullmatch(r'(IMG\d+)_(.+)', path.stem)
        if not match:
            raise ValueError(f'Unknown OCR filename: {path.name}')
        label = path.with_suffix('.txt')
        text = label.read_text(encoding='utf-8').rstrip('\r\n')
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        crop_rows.append(dict(receipt_id=match[1], field=match[2], split='val',
                              image=str(path.relative_to(ROOT)), gt=text, width=width, height=height,
                              oov=sorted(set(text)-chars), length=len(text),
                              detection_val_receipt_exists=match[1] in originals,
                              bbox_status='not_provided_in_crop_files'))
    with (ROOT / 'archives/ie_val.csv').open(encoding='utf-8-sig', newline='') as stream:
        ie_rows = list(csv.DictReader(stream))
    det_ids = set(originals)
    crop_ids = {row['receipt_id'] for row in crop_rows}
    ie_ids = {row['image_name'] for row in ie_rows}
    report = dict(
        detection=dict(receipts=len(originals), boxes=len(box_rows), classes=dict(classes),
                       text_annotations=False, class_id_to_field_mapping='not provided in downloaded ZIP or repository'),
        ocr=dict(crops=len(crop_rows), receipt_ids=len(crop_ids),
                 receipt_ids_in_detection_val=len(crop_ids & det_ids),
                 crops_with_detection_val_receipt=sum(r['detection_val_receipt_exists'] for r in crop_rows),
                 oov_samples=sum(bool(r['oov']) for r in crop_rows),
                 label_length_over25=sum(r['length']>25 for r in crop_rows)),
        ie=dict(rows=len(ie_rows), receipt_ids=len(ie_ids),
                receipt_ids_in_detection_val=len(ie_ids & det_ids), bbox_annotations=False),
        issues=issues,
        scope='Detection annotations cover key information classes, not a confirmed exhaustive text transcription.',
        linkage='Receipt-ID linkage is available. Bbox-to-transcription correspondence is NOT verified; do not assign by class order or filename guess.',
        next='Need verified class mapping and crop localization correspondence before GT-box OCR CER/EM. Full-image predictions and overlays remain possible.')
    for name, rows in [('detection_boxes',box_rows),('ocr_crops',crop_rows)]:
        with (output / f'{name}.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
