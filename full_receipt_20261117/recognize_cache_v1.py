"""Compare round2 recognition using the exact persisted pre-round2 detector crops."""
import argparse
import hashlib
import json
from pathlib import Path
from full_receipt_v1 import payload

def main():
    p = argparse.ArgumentParser()
    for key in ('cache','model','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--images',type=Path,default=Path(__file__).resolve().parent/'data/detection/val/images')
    a=p.parse_args()
    summary=json.loads((a.cache/'summary.json').read_text())
    if summary['status']!='completed':
        raise ValueError('Incomplete detection cache')
    a.output.mkdir(parents=True,exist_ok=False)
    from paddleocr import TextRecognition
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',18)
    model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(a.model),device='gpu:0',enable_mkldnn=False,cpu_threads=4)
    rows=[json.loads(s) for s in (a.cache/'receipts.jsonl').read_text().splitlines()]
    changed=total=0
    with (a.output/'receipts.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            source=a.images/row['image']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Source image changed since detection')
            canvas=Image.open(source).convert('RGB')
            draw=ImageDraw.Draw(canvas)
            for box in row['boxes']:
                result=payload(next(iter(model.predict(str(a.cache/box['crop']),batch_size=1))))
                box['predictions']['round2']=dict(text=result['rec_text'],score=float(result['rec_score']))
                total+=1
                changed+=result['rec_text']!=box['predictions']['candidate']['text']
                points=[tuple(pt) for pt in box['polygon']]
                draw.line(points+[points[0]],fill='red',width=2)
                draw.text(points[0],result['rec_text'],font=font,fill='blue',stroke_width=1,stroke_fill='white')
            canvas.save(a.output/(Path(row['image']).stem+'_round2.jpg'))
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush()
            print(row['image'],flush=True)
    receipt=dict(status='completed',n=len(rows),boxes=total,changed_texts=changed,cache_sha256=hashlib.sha256((a.cache/'receipts.jsonl').read_bytes()).hexdigest(),model_sha256=hashlib.sha256((a.model/'inference.pdiparams').read_bytes()).hexdigest(),limitation='Text changes are not accuracy; full transcriptions unavailable')
    (a.output/'summary.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
