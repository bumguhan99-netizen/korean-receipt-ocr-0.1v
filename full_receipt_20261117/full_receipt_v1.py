"""Validation-only paired OCR with one persisted detection/crop cache."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

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

def main():
    p = argparse.ArgumentParser()
    for name in ('images','detector','baseline','candidate','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.images.parent.name != 'val':
        p.error('This runner is restricted to validation until a final policy is frozen')
    a.output.mkdir(parents=True, exist_ok=False)
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from paddleocr import TextDetection, TextRecognition
    det = TextDetection(model_name='PP-OCRv5_mobile_det', model_dir=str(a.detector), device='gpu:0')
    models = {name:TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(path), device='gpu:0', enable_mkldnn=False, cpu_threads=4) for name,path in [('baseline',a.baseline),('candidate',a.candidate)]}
    paths = sorted(x for x in a.images.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'))
    if a.limit:
        paths = paths[:a.limit]
    font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 18)
    failures = []
    with (a.output/'receipts.jsonl').open('x', encoding='utf-8') as stream:
        for path in paths:
            try:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Unreadable image')
                folder = a.output/path.stem
                folder.mkdir()
                detected = payload(next(iter(det.predict(str(path)))))
                polygons = detected['dt_polys']
                row = dict(image=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), boxes=[])
                canvases = {name:Image.open(path).convert('RGB') for name in models}
                for i, polygon in enumerate(polygons):
                    crop = crop_quad(image, polygon)
                    crop_path = folder/f'{i:04d}.png'
                    if not cv2.imwrite(str(crop_path), crop):
                        raise IOError(crop_path)
                    box = dict(polygon=polygon.tolist() if hasattr(polygon,'tolist') else polygon, crop=str(crop_path.relative_to(a.output)), predictions={})
                    for name,model in models.items():
                        result = payload(next(iter(model.predict(str(crop_path), batch_size=1))))
                        box['predictions'][name] = dict(text=result['rec_text'], score=float(result['rec_score']))
                        draw = ImageDraw.Draw(canvases[name])
                        points = [tuple(pt) for pt in box['polygon']]
                        draw.line(points+[points[0]], fill='red', width=2)
                        draw.text(points[0],result['rec_text'],font=font,fill='blue',stroke_width=1,stroke_fill='white')
                    row['boxes'].append(box)
                for name,canvas in canvases.items():
                    canvas.save(folder/f'{name}.jpg')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
                print(path.name, len(row['boxes']), flush=True)
            except Exception:
                failures.append(dict(image=path.name,error=traceback.format_exc()))
    (a.output/'summary.json').write_text(json.dumps(dict(status='failed' if failures else 'completed', n=len(paths), failures=failures, limitation='Partial key-information boxes have no transcriptions; full OCR CER/precision unavailable'),ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:
        raise RuntimeError('Receipt failures recorded')

if __name__ == '__main__':
    main()
