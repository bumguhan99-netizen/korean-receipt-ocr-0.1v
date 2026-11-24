import sys,json,yaml,inspect
from pathlib import Path
import numpy as np
P=Path('/workspace/full_receipt_20261117');sys.path.insert(0,str(P/'PaddleOCR'))
from ppocr.postprocess.rec_postprocess import CTCLabelDecode as TrainingDecoder
from paddlex.inference.models.text_recognition.processors import CTCLabelDecode as RuntimeDecoder
import paddlex.inference.models.text_recognition.predictor as pred
cfg=yaml.safe_load((P/'round2/run_001/export/inference.yml').read_text())
training=TrainingDecoder(character_dict_path=str(P/'PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt'),use_space_char=True)
runtime=RuntimeDecoder(character_list=cfg['PostProcess']['character_dict'])
assert training.character==runtime.character
indices=[runtime.dict[c] for c in ['가','가','blank','가',' ','1','1','blank','1']]
scores=np.zeros((1,len(indices),len(runtime.character)),dtype='float32')
for j,i in enumerate(indices):scores[0,j,i]=1
t=runtime([scores])[0][0]
assert t=='가가 11'
source=inspect.getsource(pred)
lines=source.splitlines(); relevant=[]
for i,line in enumerate(lines):
 if 'CTCLabelDecode(' in line:relevant.extend(lines[max(0,i-4):i+6])
result=dict(status='passed',training_and_runtime_order_identical=True,base_character_count=len(cfg['PostProcess']['character_dict']),effective_character_count=len(runtime.character),blank_index=runtime.dict['blank'],space_index=runtime.dict[' '],ctc_repeat_blank_space_test=t,runtime_constructor=str(inspect.signature(RuntimeDecoder)),predictor_decoder_construction='\n'.join(relevant))
with Path('/experiment/decoder_runtime_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
