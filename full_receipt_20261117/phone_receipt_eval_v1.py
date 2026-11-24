"""Read-only smartphone receipt audit and recognition baseline; new outputs only."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'ocr_baseline/scripts'))
from metrics import measure,aggregate

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inside(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes dataset')
    return path

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)

def subset(rows,groups_path,split):
    if not groups_path:
        if split!='diagnostic':
            raise ValueError('Explicit receipt/rephotograph groups required')
        return rows
    groups=json.loads(groups_path.read_text())
    group_splits={}
    source_splits={}
    for row in rows:
        group=groups[row['receipt_id']]
        if group['split'] not in ('train','val','test') or not group['group_id']:
            raise ValueError('Invalid group entry')
        for key,seen in ((group['group_id'],group_splits),(row['source_sha256'],source_splits)):
            if key in seen and seen[key]!=group['split']:
                raise ValueError('Same receipt/source crosses splits')
            seen[key]=group['split']
    return rows if split=='diagnostic' else [r for r in rows if groups[r['receipt_id']]['split']==split]

def audit(a):
    from PIL import Image
    annotations=json.loads((a.data/'annotations.json').read_text(encoding='utf-8-sig'))
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    issues=[]
    statuses=collections.Counter()
    ids=set()
    coordinate_sizes={}
    for receipt in annotations:
        label=json.loads((a.data/'labels'/f"{receipt['id']}.json").read_text(encoding='utf-8-sig'))
        if label!=receipt:
            raise ValueError('Aggregate and receipt labels differ')
        for line in receipt['lines']:
            statuses[line['status']]+=1
            if line['id'] in ids:
                raise ValueError('Duplicate line id')
            ids.add(line['id'])
            if line['status']!='verified':
                continue
            crop=inside(a.data,line['crop'])
            coordinate_relative=line.get('coordinate_image')
            if not coordinate_relative:
                repair=receipt.get('crop_coordinate_repair',{})
                if repair.get('rotation_degrees')!=0 or not repair.get('coordinate_image'):
                    raise ValueError('Missing ambiguous coordinate frame')
                coordinate_relative=repair['coordinate_image']
            coordinate=inside(a.data,coordinate_relative)
            verified=a.data/'verified_labels'/f"{line['id']}.txt"
            if verified.read_text(encoding='utf-8')!=line['text']:
                raise ValueError('Verified transcription differs')
            with Image.open(crop) as image:
                image.load()
                dimensions=image.size
            if coordinate_relative not in coordinate_sizes:
                with Image.open(coordinate) as image:
                    coordinate_sizes[coordinate_relative]=image.size
            width,height=coordinate_sizes[coordinate_relative]
            x1,y1,x2,y2=line['bbox']
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise ValueError('Box outside coordinate image')
            rotation=line.get('crop_rotation_degrees',0)
            expected=(y2-y1,x2-x1) if rotation in (90,270) else (x2-x1,y2-y1)
            if dimensions!=expected:
                issues.append(dict(line_id=line['id'],reason='crop_size_differs_from_box',actual=dimensions,expected=expected))
            if 'coordinate_image' not in line:
                from PIL import ImageChops
                with Image.open(coordinate) as source, Image.open(crop) as saved:
                    expected_image=source.crop(line['bbox']).convert('RGB').rotate(rotation,expand=True)
                    if expected_image.size!=saved.size or ImageChops.difference(expected_image,saved.convert('RGB')).getbbox():
                        raise ValueError('Fallback coordinate frame not confirmed by exact crop pixels')
            rows.append(dict(receipt_id=receipt['id'],source_sha256=receipt['source_sha256'],line_id=line['id'],crop=line['crop'],crop_sha256=sha(crop),text=line['text'],bbox=line['bbox'],coordinate_image=coordinate_relative,rotation=rotation))
    with (a.output/'manifest.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(a.output/'audit.json',dict(status='passed' if not issues else 'needs_inspection',receipts=len(annotations),statuses=dict(statuses),verified=len(rows),issues=issues,annotations_sha256=sha(a.data/'annotations.json'),full_coverage_certified=False,split_status='unassigned; requires receipt/rephotograph grouping before a future training split'))

def evaluate(a):
    from paddleocr import TextRecognition
    from full_receipt_v1 import payload
    report=json.loads((a.manifest.parent/'audit.json').read_text())
    if report['status']!='passed':
        raise ValueError('Audit issues must be resolved or explicitly versioned')
    if sha(a.data/'annotations.json')!=report['annotations_sha256']:
        raise ValueError('Annotations changed since audit')
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines()]
    rows=subset(rows,a.groups,a.split)
    if a.split=='test':
        raise ValueError('Final test requires a separately frozen post-tuning protocol; this tool is pre-tuning only')
    if not rows:
        raise ValueError('Empty subset')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(model={f:sha(a.model/f) for f in ('inference.json','inference.pdiparams','inference.yml')},manifest_sha256=sha(a.manifest),split=a.split,normalization='NFC',spaces='preserved',punctuation='preserved',n=len(rows),groups_sha256=sha(a.groups) if a.groups else None)
    write(a.output/'protocol.json',protocol)
    model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(a.model),device=a.device,enable_mkldnn=False,cpu_threads=4)
    scored=[]
    started=time.monotonic()
    with (a.output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            crop=inside(a.data,row['crop'])
            if sha(crop)!=row['crop_sha256']:
                raise ValueError('Audited crop changed')
            result=payload(next(iter(model.predict(str(crop),batch_size=1))))
            score=dict(receipt_id=row['receipt_id'],line_id=row['line_id'],ground_truth=row['text'],prediction=result['rec_text'],**measure(row['text'],result['rec_text']))
            scored.append(score)
            stream.write(json.dumps(score,ensure_ascii=False)+'\n')
            stream.flush()
    write(a.output/'summary.json',dict(status='completed',**aggregate(scored),seconds=time.monotonic()-started,scope='verified recognition crops; not full receipt accuracy',amount_em=None,amount_reason='No structured field GT',future_holdout_warning='Any diagnostic images inspected here must not later be described as an untouched test set'))
    write(a.output/'by_receipt.json',{rid:aggregate([r for r in scored if r['receipt_id']==rid]) for rid in sorted({r['receipt_id'] for r in scored})})

def full(a):
    import cv2
    from paddleocr import TextDetection,TextRecognition
    from full_receipt_v1 import payload,crop_quad
    from partial_matching_v1 import match
    if a.split=='test':
        raise ValueError('Pre-tuning script cannot consume final test')
    audit_report=json.loads((a.manifest.parent/'audit.json').read_text())
    if audit_report['status']!='passed':
        raise ValueError('Audit must pass')
    if sha(a.data/'annotations.json')!=audit_report['annotations_sha256']:
        raise ValueError('Annotations changed since audit')
    rows=[json.loads(s) for s in a.manifest.read_text().splitlines()]
    rows=subset(rows,a.groups,a.split)
    if not rows:
        raise ValueError('Empty subset')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'coordinate_images.json',{relative:sha(inside(a.data,relative)) for relative in sorted({r['coordinate_image'] for r in rows})})
    write(a.output/'protocol.json',dict(model_sha256=sha(a.model/'inference.pdiparams'),detector_sha256=sha(a.detector/'inference.pdiparams'),manifest_sha256=sha(a.manifest),iou=0.5,matching='maximum cardinality then IoU; axis-aligned bounds in coordinate_image',missed_gt_prediction='',unmatched_predictions='unscored',normalization='NFC',split=a.split))
    detector=TextDetection(model_name='PP-OCRv5_mobile_det',model_dir=str(a.detector),device=a.device)
    recognizer=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(a.model),device=a.device,enable_mkldnn=False,cpu_threads=4)
    grouped=collections.defaultdict(list)
    for row in rows:
        grouped[(row['receipt_id'],row['coordinate_image'])].append(row)
    scored=[]
    matched=missing=unscored=0
    with (a.output/'partial_predictions.jsonl').open('x',encoding='utf-8') as stream:
        for (rid,relative),gt in sorted(grouped.items()):
            path=inside(a.data,relative)
            image=cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable coordinate image')
            polygons=payload(next(iter(detector.predict(str(path)))))['dt_polys']
            bounds=[[float(min(p[0] for p in polygon)),float(min(p[1] for p in polygon)),float(max(p[0] for p in polygon)),float(max(p[1] for p in polygon))] for polygon in polygons]
            result=match([r['bbox'] for r in gt],bounds)
            predictions={}
            for gi,pi,overlap in result['pairs']:
                crop=crop_quad(image,polygons[pi])
                rec=payload(next(iter(recognizer.predict(crop,batch_size=1))))
                predictions[gi]=(rec['rec_text'],overlap)
            for i,row in enumerate(gt):
                prediction,overlap=predictions.get(i,('',0.0))
                item=dict(receipt_id=rid,line_id=row['line_id'],ground_truth=row['text'],prediction=prediction,matched=i in predictions,iou=overlap,**measure(row['text'],prediction))
                scored.append(item)
                stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            stream.flush()
            matched+=len(result['pairs'])
            missing+=len(result['missing_gt'])
            unscored+=len(result['unscored_predictions'])
            print(rid,flush=True)
    write(a.output/'summary.json',dict(status='completed',**aggregate(scored),matched_gt=matched,missing_gt=missing,partial_gt_recall=matched/(matched+missing) if matched+missing else None,unscored_predictions=unscored,precision=None,scope='verified boxes only, aligned coordinate images; missing GT counted as deletions; not complete receipt accuracy'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','evaluate','full'])
    p.add_argument('--data',type=Path,default=PROJECT/'dataset/receipt_labeled')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--detector',type=Path)
    p.add_argument('--groups',type=Path)
    p.add_argument('--split',choices=['diagnostic','val','test'],default='diagnostic')
    p.add_argument('--device',default='gpu:0')
    a=p.parse_args()
    if a.stage in ('evaluate','full') and (not a.manifest or not a.model):
        p.error('--manifest and --model are required')
    if a.stage=='full' and not a.detector:
        p.error('--detector is required')
    {'audit':audit,'evaluate':evaluate,'full':full}[a.stage](a)

if __name__=='__main__':
    main()
