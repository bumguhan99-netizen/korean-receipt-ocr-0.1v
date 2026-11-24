"""GT-free receipt region inference with frozen recognition weights.
No labels, GT boxes, matching thresholds, or per-receipt overrides are used.
Validated on the 304-receipt diagnostic corpus; not an unseen test claim.
"""
import argparse,hashlib,json
from pathlib import Path
POLICY = dict(limit_side_len=1920,limit_type="max",thresh=0.3,box_thresh=0.45,unclip_ratio=1.2)
MODEL_SHA256 = "10e0e6b65bce1582339440abcf844e1c1e40a1d6357c5722a177c0dabce8a155"
def payload(result):
    value = result.json
    if isinstance(value, str):
        value = json.loads(value)
    return value.get('res', value)

def crop_quad(image, polygon):
    import cv2
    import numpy as np
    pts = np.asarray(polygon, dtype='float32')
    if pts.shape != (4, 2):
        raise ValueError('Expected quadrilateral')
    width = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])), 1)
    height = max(int(np.linalg.norm(pts[0]-pts[3])), int(np.linalg.norm(pts[1]-pts[2])), 1)
    matrix = cv2.getPerspectiveTransform(pts, np.array([[0,0],[width,0],[width,height],[0,height]], dtype='float32'))
    crop = cv2.warpPerspective(image, matrix, (width,height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.rot90(crop) if height / width >= 1.5 else crop

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

class RegionOCR:
    def __init__(self, detector_dir, recognizer_dir, device="gpu:0"):
        from paddleocr import TextDetection, TextRecognition
        recognizer_dir=Path(recognizer_dir)
        digest=hashlib.sha256((recognizer_dir/"inference.pdiparams").read_bytes()).hexdigest()
        if digest!=MODEL_SHA256:
            raise ValueError("Frozen recognition model hash mismatch")
        self.detector=TextDetection(model_name="PP-OCRv5_mobile_det",model_dir=str(detector_dir),device=device,**POLICY)
        self.recognizer=TextRecognition(model_name="korean_PP-OCRv5_mobile_rec",model_dir=str(recognizer_dir),device=device,enable_mkldnn=False,cpu_threads=4)

    def predict(self, image_path):
        import cv2
        image=cv2.imread(str(image_path))
        if image is None: raise ValueError("Unreadable image")
        raw=payload(next(iter(self.detector.predict(str(image_path)))))
        detections=[]
        for idx,polygon in enumerate(raw["dt_polys"]):
            polygon=polygon.tolist() if hasattr(polygon,"tolist") else polygon
            crop=crop_quad(image,polygon)
            rec=payload(next(iter(self.recognizer.predict(crop,batch_size=1))))
            detections.append(dict(index=idx,polygon=polygon,bbox=bounds(polygon),text=rec["rec_text"],score=float(rec["rec_score"])))
        return dict(image=str(image_path),raw_detections=detections,regions=merge_fragments(detections),policy=POLICY,model_sha256=MODEL_SHA256)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--image",type=Path,required=True);p.add_argument("--detector",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="gpu:0");a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=RegionOCR(a.detector,a.model,a.device).predict(a.image)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
