"""Fixed-recognizer region experiments. Inference never receives annotations."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
P=Path('/workspace'); F=P/'full_receipt_20261117'; B=F/'phone_baseline'; DATA=P/'dataset/receipt_labeled'
sys.path[:0]=[str(F),str(P/'ocr_baseline/scripts')]
from metrics import measure,aggregate
from partial_matching_v1 import match,iou
from full_receipt_v1 import payload,crop_quad
MODEL=F/'round2/run_001/export'; DET=F/'models/PP-OCRv5_mobile_det'
EXPECTED='10e0e6b65bce1582339440abcf844e1c1e40a1d6357c5722a177c0dabce8a155'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def write(p,v):
 with p.open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2)
def selected():
 manifest=rows(B/'audit_v2/manifest.jsonl'); images=sorted({(r['receipt_id'],r['coordinate_image']) for r in manifest})
 # Deterministic development subset, disjoint by source hash. Already-diagnosed corpus is NOT a clean test.
 sources={r['receipt_id']:r['source_sha256'] for r in manifest}
 devgroups=set(sorted(set(sources.values()),key=lambda s:hashlib.sha256(('region-v1:'+s).encode()).hexdigest())[:48])
 return manifest,images,{rid for rid,s in sources.items() if s in devgroups}
def bounds(poly): return [min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)]
def merge_fragments(ds):
 # Only close, same-baseline fragments; never look at GT. Preserve source indices for audit.
 out=[]; used=set()
 for i in sorted(range(len(ds)),key=lambda i:(ds[i]['bbox'][1],ds[i]['bbox'][0])):
  if i in used: continue
  chain=[i]; b=list(ds[i]['bbox']); used.add(i)
  while True:
   candidates=[]
   for j,d in enumerate(ds):
    if j in used: continue
    c=d['bbox']; h=min(b[3]-b[1],c[3]-c[1]); gap=c[0]-b[2]
    vy=max(0,min(b[3],c[3])-max(b[1],c[1]))/max(h,1)
    ratio=max(b[3]-b[1],c[3]-c[1])/max(h,1)
    left=''.join(ds[chain[-1]]['text'].split()); right=''.join(d['text'].split())
    barcode=left.isdigit() and len(left)>=12 and any(t.isdigit() for t in right) and len(right)<=9
    short=(len(left)<=2 or len(right)<=2)
    if vy>=.75 and ratio<=1.6 and -.1*h<=gap<=(2.8 if barcode else .65)*h and (barcode or short): candidates.append((gap,j))
   if not candidates: break
   j=min(candidates)[1]; used.add(j); chain.append(j); c=ds[j]['bbox']; b=[min(b[0],c[0]),min(b[1],c[1]),max(b[2],c[2]),max(b[3],c[3])]
  if len(chain)==1: out.append(dict(ds[i],source_indices=chain))
  else: out.append(dict(bbox=b,polygon=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],text=' '.join(ds[j]['text'] for j in chain),score=min(ds[j]['score'] for j in chain),source_indices=chain,merged=True))
 return out
def infer(a):
 import cv2
 from paddleocr import TextDetection,TextRecognition
 manifest,images,dev=selected(); images=[v for v in images if a.subset=='all' or (v[0] in dev)==(a.subset=='dev')]
 assert sha(MODEL/'inference.pdiparams')==EXPECTED
 a.output.mkdir(parents=True,exist_ok=False)
 cfg=dict(limit_side_len=a.resolution,limit_type='max',thresh=.3,box_thresh=a.box_thresh,unclip_ratio=a.unclip) if a.resolution else {}
 protocol=dict(config=cfg,model={n:sha(MODEL/n) for n in ('inference.pdiparams','inference.json','inference.yml')},detector_sha256=sha(DET/'inference.pdiparams'),manifest_sha256=sha(B/'audit_v2/manifest.jsonl'),subset=a.subset,dev_receipts=sorted(dev),image_hashes={rel:sha(DATA/rel) for _,rel in images},scoring='Original IoU>=0.5 maximum-cardinality matching; NFC; missing=empty; unannotated detections unscored',script_sha256=sha(Path(__file__)))
 write(a.output/'protocol.json',protocol)
 det=TextDetection(model_name='PP-OCRv5_mobile_det',model_dir=str(DET),device='gpu:0',**cfg)
 rec=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(MODEL),device='gpu:0',enable_mkldnn=False,cpu_threads=4)
 started=time.monotonic()
 for n,(rid,rel) in enumerate(images):
  image=cv2.imread(str(DATA/rel)); raw=payload(next(iter(det.predict(str(DATA/rel)))))
  ds=[]
  for idx,poly in enumerate(raw['dt_polys']):
   poly=poly.tolist() if hasattr(poly,'tolist') else poly
   crop=crop_quad(image,poly); pred=payload(next(iter(rec.predict(crop,batch_size=1))))
   ds.append(dict(index=idx,polygon=poly,bbox=bounds(poly),text=pred['rec_text'],score=float(pred['rec_score'])))
  write(a.output/(rid+'.json'),dict(receipt_id=rid,coordinate_image=rel,detections=ds))
  if (n+1)%10==0: print(json.dumps(dict(completed=n+1,total=len(images),seconds=round(time.monotonic()-started,1))),flush=True)
 assert sha(MODEL/'inference.pdiparams')==EXPECTED
 write(a.output/'completed.json',dict(images=len(images),seconds=time.monotonic()-started))
def score(a):
 manifest,images,dev=selected(); bykey=collections.defaultdict(list)
 for r in manifest: bykey[(r['receipt_id'],r['coordinate_image'])].append(r)
 old={r['line_id']:r for r in rows(B/'aligned_partial_v1/partial_predictions.jsonl')}; crop={r['line_id']:r for r in rows(B/'crops_diagnostic_v1/predictions.jsonl')}
 if a.baseline: caches=[load(p) for p in (B/'aligned_partial_v1').glob('*/detection_cache.json')]
 else:
  assert (a.input/'completed.json').exists()
  caches=[load(p) for p in a.input.glob('r*.json')]
 scored=[]; unscored=0
 for c in caches:
  if a.subset!='all' and ((c['receipt_id'] in dev)!=(a.subset=='dev')): continue
  gt=bykey[(c['receipt_id'],c['coordinate_image'])]; ds=merge_fragments(c['detections']) if a.merge else c['detections']
  result=match([r['bbox'] for r in gt],[d['bbox'] for d in ds]); pairs={i:(j,v) for i,j,v in result['pairs']}; unscored+=len(result['unscored_predictions'])
  for i,r in enumerate(gt):
   j,v=pairs.get(i,(None,0)); text=ds[j]['text'] if j is not None else ''; before=old[r['line_id']]
   scored.append(dict(line_id=r['line_id'],receipt_id=r['receipt_id'],prediction=text,matched=j is not None,iou=v,before_missing=not before['matched'],before_crop_exact_full_wrong=bool(before['matched'] and crop[r['line_id']]['exact_match'] and not before['exact_match']),before_exact=before['exact_match'],**measure(r['text'],text)))
 def stats(rr): return dict(**aggregate(rr),missing_gt=sum(not r['matched'] for r in rr))
 summary=dict(status='completed',**stats(scored),unscored_predictions=unscored,merge=a.merge,subset=a.subset,images=len(set(r['receipt_id'] for r in scored)),recovered_missing=sum(r['before_missing'] and r['matched'] for r in scored),recovered_missing_exact=sum(r['before_missing'] and r['exact_match'] for r in scored),recovered_624_exact=sum(r['before_crop_exact_full_wrong'] and r['exact_match'] for r in scored),regressed_exact=sum(r['before_exact'] and not r['exact_match'] for r in scored),improved_exact=sum(not r['before_exact'] and r['exact_match'] for r in scored),development=stats([r for r in scored if r['receipt_id'] in dev]),confirmation=stats([r for r in scored if r['receipt_id'] not in dev]))
 a.output.mkdir(parents=True,exist_ok=False); write(a.output/'summary.json',summary)
 with (a.output/'predictions.jsonl').open('x',encoding='utf-8') as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['infer','score']);p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--subset',choices=['all','dev','confirm'],default='all');p.add_argument('--resolution',type=int,default=1920);p.add_argument('--unclip',type=float,default=1.2);p.add_argument('--box-thresh',type=float,default=.45);p.add_argument('--merge',action='store_true');p.add_argument('--baseline',action='store_true');a=p.parse_args(); {'infer':infer,'score':score}[a.stage](a)
