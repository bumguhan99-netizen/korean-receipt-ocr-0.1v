"""Paired mobile/server comparison with fixed recognition and scoring."""
import sys,json,time,argparse,hashlib,collections,importlib.metadata
from pathlib import Path
import numpy as np
import cv2,paddle
from paddleocr import TextDetection,TextRecognition
sys.path.insert(0,'/experiment')
from region_experiment import selected,sha,write,MODEL,EXPECTED,DATA,F,B,bounds,score
from full_receipt_v1 import payload,crop_quad
OUT=Path('/experiment/detector_comparison_v2');OUT.mkdir(exist_ok=False)
manifest,images,dev=selected(); assert len(images)==304 and len(manifest)==8343
cfg=dict(limit_side_len=1920,limit_type='max',thresh=.3,box_thresh=.45,unclip_ratio=1.2)
paths=dict(mobile=F/'models/PP-OCRv5_mobile_det',server=Path('/experiment/server_detector_download_v2'))
assert sha(MODEL/'inference.pdiparams')==EXPECTED
protocol=dict(config=cfg,recognizer_sha256=EXPECTED,recognizer_config_sha256=sha(MODEL/'inference.yml'),detector_hashes={k:{n:sha(p/n) for n in ('inference.json','inference.pdiparams','inference.yml')} for k,p in paths.items()},manifest_sha256=sha(B/'audit_v2/manifest.jsonl'),images={rel:sha(DATA/rel) for _,rel in images},gpu=paddle.device.cuda.get_device_name(),packages={p:importlib.metadata.version(p) for p in ('paddlepaddle-gpu','paddleocr','paddlex')},batch_size=1,threads=4,warmup=3,timing='CUDA synchronized wall time; detector includes preprocess/inference/postprocess, excludes disk image decode; full includes decode+detector+crop+recognition; no matching/scoring/JSON I/O; models loaded once; alternating order per receipt; no fragment merge',repeated_detection_passes=3,selection='fixed common settings from previous mobile experiment; no server tuning',scope='304 aligned diagnostic receipts with partial GT, not unseen test',script_sha256=sha(Path(__file__)))
write(OUT/'protocol.json',protocol)
init={}; detectors={}
for name,path in paths.items():
 t=time.perf_counter();detectors[name]=TextDetection(model_name='PP-OCRv5_'+name+'_det',model_dir=str(path),device='gpu:0',enable_mkldnn=False,cpu_threads=4,**cfg);paddle.device.cuda.synchronize();init[name]=time.perf_counter()-t
rec=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(MODEL),device='gpu:0',enable_mkldnn=False,cpu_threads=4)
effective={name:det._extra_init_args for name,det in detectors.items()}
assert all(all(effective[name][key]==value for key,value in cfg.items()) for name in detectors)
write(OUT/'effective_config.json',effective)
for name in paths:(OUT/name).mkdir()
# Same warm-up inputs for both detectors; all outside measured intervals.
for rid,rel in images[:3]:
 im=cv2.imread(str(DATA/rel))
 for det in detectors.values():payload(next(iter(det.predict(im,batch_size=1))))
 r=next(r for r in manifest if r['receipt_id']==rid)
 payload(next(iter(rec.predict(str(DATA/r['crop']),batch_size=1))))
paddle.device.cuda.synchronize()
timings=[];started=time.perf_counter();replay_diffs=[]
for ix,(rid,rel) in enumerate(images):
 t=time.perf_counter();image=cv2.imread(str(DATA/rel));io=time.perf_counter()-t
 for name in (('mobile','server') if ix%2==0 else ('server','mobile')):
  paddle.device.cuda.synchronize();t=time.perf_counter();raw=payload(next(iter(detectors[name].predict(image,batch_size=1))));paddle.device.cuda.synchronize();dt=time.perf_counter()-t
  t=time.perf_counter();ds=[]
  for idx,poly in enumerate(raw['dt_polys']):
   poly=poly.tolist() if hasattr(poly,'tolist') else poly
   crop=crop_quad(image,poly);rr=payload(next(iter(rec.predict(crop,batch_size=1))))
   ds.append(dict(index=idx,polygon=poly,bbox=bounds(poly),text=rr['rec_text'],score=float(rr['rec_score'])))
  paddle.device.cuda.synchronize();rt=time.perf_counter()-t
  write(OUT/name/(rid+'.json'),dict(receipt_id=rid,coordinate_image=rel,detections=ds))
  timings.append(dict(receipt_id=rid,model=name,decode_s=io,detector_s=dt,crop_recognition_s=rt,pipeline_s=io+dt+rt,detections=len(ds)))
  if name=='mobile':
   prev=json.loads((Path('/experiment/final_1920')/(rid+'.json')).read_text())
   if [(d['bbox'],d['text']) for d in prev['detections']]!=[(d['bbox'],d['text']) for d in ds]:
    replay_diffs.append(rid)
    raise AssertionError('Tuned mobile reference mismatch: '+rid)
 if (ix+1)%20==0:print(json.dumps(dict(stage='paired_accuracy',images=ix+1,total=304,seconds=round(time.perf_counter()-started,1))),flush=True)
for name in paths:
 write(OUT/name/'completed.json',dict(images=len(images),seconds=sum(r['pipeline_s'] for r in timings if r['model']==name)))
 score(argparse.Namespace(baseline=False,input=OUT/name,output=OUT/(name+'_scores'),subset='all',merge=False))
write(OUT/'pipeline_timings.json',timings);write(OUT/'baseline_replay.json',dict(differing_receipts=replay_diffs,checked=304))
# Detection-only repeats on all 304 images. Alternate order to reduce cache/order bias.
repeats=[]
for rep in range(3):
 for ix,(rid,rel) in enumerate(images):
  image=cv2.imread(str(DATA/rel))
  for name in (('mobile','server') if (ix+rep)%2==0 else ('server','mobile')):
   paddle.device.cuda.synchronize();t=time.perf_counter();raw=payload(next(iter(detectors[name].predict(image,batch_size=1))));paddle.device.cuda.synchronize();elapsed=time.perf_counter()-t
   repeats.append(dict(repeat=rep,receipt_id=rid,model=name,seconds=elapsed,detections=len(raw['dt_polys'])))
  if (ix+1)%100==0:print(json.dumps(dict(stage='detector_speed',repeat=rep+1,images=ix+1)),flush=True)
write(OUT/'detector_repeats.json',repeats)
def describe(vals):return dict(mean_s=float(np.mean(vals)),median_s=float(np.median(vals)),p95_s=float(np.percentile(vals,95)),total_s=float(sum(vals)),n=len(vals))
summary={}
for name in paths:
 rr=[r for r in timings if r['model']==name];dd=[r for r in repeats if r['model']==name]
 summary[name]=dict(accuracy=json.loads((OUT/(name+'_scores')/'summary.json').read_text()),load_seconds=init[name],detector=describe([r['seconds'] for r in dd]),pipeline=describe([r['pipeline_s'] for r in rr]),crop_recognition=describe([r['crop_recognition_s'] for r in rr]),detection_count=sum(r['detections'] for r in rr),detector_pass_totals_s=[sum(r['seconds'] for r in dd if r['repeat']==rep) for rep in range(3)])
assert sha(MODEL/'inference.pdiparams')==EXPECTED
write(OUT/'summary.json',dict(status='completed',models=summary,mobile_replay_differences=replay_diffs))
print('COMPARISON COMPLETE',flush=True)
