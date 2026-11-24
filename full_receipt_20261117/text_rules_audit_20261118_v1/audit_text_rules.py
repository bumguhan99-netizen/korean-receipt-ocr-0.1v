import json,csv,sys,collections,unicodedata as U,hashlib,html
from pathlib import Path
import yaml
P=Path(r'D:\ssafy\ocr\mogumogu');F=P/'full_receipt_20261117';B=F/'phone_baseline'
sys.path.insert(0,str(P/'ocr_baseline/scripts'))
from metrics import measure,aggregate,distance
O=Path(__file__).resolve().parents[1]/'outputs/text_rules_audit_20261118_v1';O.mkdir(parents=True,exist_ok=False)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
model=F/'round2/run_001/export'; cfg=yaml.safe_load((model/'inference.yml').read_text(encoding='utf-8')); train=yaml.safe_load((F/'round2/run_001/train.yml').read_text(encoding='utf-8'))
dictionary=F/'PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt'; chars=dictionary.read_text(encoding='utf-8').splitlines(); exported=cfg['PostProcess']['character_dict']; expected=chars+([' '] if train['Global']['use_space_char'] else [])
vocab=set(exported)|{' '}; manifest=rows(B/'audit_v2/manifest.jsonl');truth={r['line_id']:r['text'] for r in manifest}
sources={'crop':B/'crops_diagnostic_v1/predictions.jsonl','before':B/'aligned_partial_v1/partial_predictions.jsonl','after_detector':O.parent/'region_improvement_20261118_v1'/'final_plain_missing_placeholder','after_merge':O.parent/'region_improvement_20261118_v1/evaluation/predictions.jsonl'}
sources['after_detector']=Path(__file__).parent/'final_plain/predictions.jsonl'
normal=lambda t:U.normalize('NFC',t)
strip=lambda t:''.join(c for c in normal(t) if not c.isspace())
result={};csvrows=[];oov_counts=collections.Counter();oov_lines=collections.defaultdict(list); whitespace=collections.Counter();gt_changed=[];nfkc_changed=[]
for key,t in truth.items():
 if normal(t)!=t:gt_changed.append(key)
 if U.normalize('NFKC',t)!=normal(t):nfkc_changed.append(key)
 for c in normal(t):
  if c.isspace():whitespace[f'U+{ord(c):04X} {U.name(c,"UNKNOWN")}']+=1
  elif c not in vocab:oov_counts[c]+=1
 for c in set(normal(t))-vocab:
  if not c.isspace():oov_lines[c].append(key)
for name,path in sources.items():
 rr=rows(path);assert {r['line_id'] for r in rr}==set(truth)
 keep=[];ignore=[];ascii_ignore=[];only=[];changed_pred=[];unexpected=collections.Counter();raw_edits=0;raw_exact=0;pred_ws=collections.Counter();oov_group=[];covered_group=[]
 for r in rr:
  key=r['line_id'];g=truth[key];p=r['prediction'];a=measure(g,p);b=measure(strip(g),strip(p));c=measure(normal(g).replace(' ',''),normal(p).replace(' ',''))
  assert a['char_edits']==r['char_edits'] and a['exact_match']==r['exact_match']
  keep.append(a);ignore.append(b);ascii_ignore.append(c);raw_edits+=distance(g,p);raw_exact+=g==p
  if normal(p)!=p:changed_pred.append(key)
  if not a['exact_match'] and b['exact_match']:only.append(key)
  for ch in normal(p):
   if ch.isspace():pred_ws[f'U+{ord(ch):04X} {U.name(ch,"UNKNOWN")}']+=1
   elif ch not in vocab:unexpected[ch]+=1
  oov=any(ch not in vocab and not ch.isspace() for ch in normal(g))
  (oov_group if oov else covered_group).append(a)
  csvrows.append(dict(condition=name,line_id=key,ground_truth=g,prediction=p,keep_edits=a['char_edits'],keep_gt_chars=a['gt_chars'],ignore_edits=b['char_edits'],ignore_gt_chars=b['gt_chars'],keep_exact=a['exact_match'],ignore_exact=b['exact_match'],whitespace_only=int(not a['exact_match'] and b['exact_match']),gt_oov=oov))
 result[name]=dict(source=str(path),source_sha256=sha(path),keep=aggregate(keep),ignore_whitespace=aggregate(ignore),ignore_ascii_space=aggregate(ascii_ignore),whitespace_only_n=len(only),whitespace_only_examples=[dict(line_id=k,gt=truth[k],prediction=next(r['prediction'] for r in rr if r['line_id']==k)) for k in only[:12]],raw_edits=raw_edits,raw_exact=raw_exact,nfc_changed_predictions=len(changed_pred),prediction_oov=dict(unexpected),prediction_whitespace=dict(pred_ws),gt_oov_group=aggregate(oov_group),gt_covered_group=aggregate(covered_group))
duplicates={c:n for c,n in collections.Counter(exported).items() if n>1}
audit=dict(training_dict_sha256=sha(dictionary),training_dict_count=len(chars),exported_count=len(exported),ordered_match_export_vs_train=exported==expected,ordered_match_export_vs_file_without_space=exported==chars,use_space_char=train['Global']['use_space_char'],space_indices=[i for i,c in enumerate(exported) if c==' '],duplicate_export_entries=duplicates,multichar_entries=[c for c in exported if len(c)!=1],gt_nfc_changed=len(gt_changed),gt_nfkc_changed=len(nfkc_changed),gt_whitespace=dict(whitespace),gt_oov_lines=len(set(k for ids in oov_lines.values() for k in ids)),gt_oov=[dict(character=c,codepoint=f'U+{ord(c):04X}',name=U.name(c,'UNKNOWN'),occurrences=n,line_count=len(oov_lines[c]),examples=oov_lines[c][:8],nfkc=U.normalize('NFKC',c)) for c,n in oov_counts.most_common()],model_sha256=sha(model/'inference.pdiparams'),policy='NFC; whitespace-ignore removes all Unicode isspace characters only; punctuation preserved; no inference rerun; ignore CER denominator excludes whitespace; WER after removal is not reported')
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
write(O/'audit.json',dict(dictionary=audit,conditions=result))
with (O/'line_scores.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
print(json.dumps(audit,ensure_ascii=False,indent=2))
for name,r in result.items():print(name,json.dumps(dict(keep=r['keep'],ignore=r['ignore_whitespace'],whitespace_only=r['whitespace_only_n'],nfc_changed_pred=r['nfc_changed_predictions'],oov=r['prediction_oov']),ensure_ascii=False))
