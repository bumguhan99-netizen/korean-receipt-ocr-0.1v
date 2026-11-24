"""Join completed baseline tracks and produce a source-backed technical readout."""
import collections
import json
from pathlib import Path
import unicodedata
import yaml
from phone_receipt_eval_v2 import aggregate,write

root=Path(__file__).resolve().parent
base=root/'phone_baseline'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]
crop_summary=read(base/'crops_diagnostic_v1/summary.json')
full_summary=read(base/'aligned_partial_v1/summary.json')
raw_summary=read(base/'raw_preview_v2/summary.json')
if any(r['status']!='completed' for r in (crop_summary,full_summary,raw_summary)):
    raise ValueError('All evaluation tracks must finish')
manifest=rows(base/'audit_v2/manifest.jsonl')
crop={r['line_id']:r for r in rows(base/'crops_diagnostic_v1/predictions.jsonl')}
full={r['line_id']:r for r in rows(base/'aligned_partial_v1/partial_predictions.jsonl')}
if set(crop)!=set(full) or set(crop)!={r['line_id'] for r in manifest} or len(crop)!=8343:
    raise ValueError('Track sample sets differ')
characters=set(yaml.safe_load((root/'round2/run_001/export/inference.yml').read_text())['PostProcess']['character_dict'])|{' '}
out=base/'analysis_v2'
out.mkdir(exist_ok=False)
buckets=collections.defaultdict(list)
cross=collections.Counter()
examples=collections.defaultdict(list)
for row in manifest:
    c,f=crop[row['line_id']],full[row['line_id']]
    text=unicodedata.normalize('NFC',row['text'])
    length='length_1_10' if len(text)<=10 else 'length_11_25' if len(text)<=25 else 'length_over25'
    x1,y1,x2,y2=row['bbox']
    height=(x2-x1) if row['rotation'] in (90,270) else (y2-y1)
    for key in [length,'oov' if set(text)-characters else 'dictionary_covered','height_le20' if height<=20 else 'height_gt20','rotated_crop' if row['rotation'] else 'unrotated_crop']:
        buckets[key].append(c)
    if not f['matched']:
        category='unmatched_gt_crop_correct' if c['exact_match'] else 'unmatched_gt_crop_incorrect'
    elif c['exact_match'] and f['exact_match']:
        category='both_exact'
    elif c['exact_match']:
        category='crop_exact_full_incorrect'
    elif f['exact_match']:
        category='crop_incorrect_full_exact'
    else:
        category='both_incorrect_matched'
    cross[category]+=1
    if len(examples[category])<8:
        examples[category].append(dict(line_id=row['line_id'],receipt_id=row['receipt_id'],gt=row['text'],crop_prediction=c['prediction'],full_prediction=f['prediction'],iou=f['iou'],crop_path=row['crop']))
bucket_results={key:aggregate(value) for key,value in sorted(buckets.items())}
matched_metrics=aggregate([r for r in full.values() if r['matched']])
missed_metrics=aggregate([r for r in full.values() if not r['matched']])
worst=sorted(crop.values(),key=lambda r:(r['char_edits']/max(1,r['gt_chars']),r['char_edits']),reverse=True)[:30]
whitespace_only=[r for r in crop.values() if not r['exact_match'] and ''.join(r['ground_truth_norm'].split())==''.join(r['prediction_norm'].split())]
write(out/'analysis.json',dict(crop=crop_summary,partial=full_summary,raw=raw_summary,cross_categories=dict(cross),buckets=bucket_results,matched_only=matched_metrics,missed_only=missed_metrics,examples=dict(examples),worst_crops=worst,whitespace_only_errors=len(whitespace_only)))
lines=['스마트폰 촬영 영수증 튜닝 전 진단 평가 결과','',
    '기준 모델: KORIE 2차 최종 채택 모델(round2/run_001/export). 스마트폰 데이터 학습은 수행하지 않았다.',
    '데이터: receipt_labeled 304장, verified 8,343줄. excluded 1,732줄은 정답 집계에서 제외했다.',
    '이번 전체 집합 평가는 diagnostic이다. 이 결과를 분석·활용한 집합을 이후 미사용 최종 test로 주장하지 않는다.',
    '',
    f"GT crop: CER {crop_summary['cer_pct']:.4f}%, 문자열 EM {crop_summary['exact_match_pct']:.4f}%, 숫자·기호 EM {crop_summary['numeric_symbol_only_exact_pct']:.4f}%.",
    f"Aligned 부분 GT 전체 OCR: CER {full_summary['cer_pct']:.4f}%, 문자열 EM {full_summary['exact_match_pct']:.4f}%.",
    f"IoU>=0.5 일대일 매칭 {full_summary['matched_gt']}줄, 미매칭 GT {full_summary['missing_gt']}줄, 부분 GT 매칭률 {100*full_summary['partial_gt_recall']:.2f}%.",
    f"미매칭 예측 {full_summary['unscored_predictions']}개는 정답 미제공 범위이므로 오검출로 집계하지 않았다.",
    f"매칭된 GT만의 CER {matched_metrics['cer_pct']:.4f}%, EM {matched_metrics['exact_match_pct']:.4f}%. 미매칭 GT의 총 삭제 문자 수 {missed_metrics['char_edits']}.",
    '위 수치는 완전한 원본 영수증 OCR 정확도/검출 precision이 아니다. 박스 병합·분할, 부분 잘림, 좌표 근사도 IoU 매칭에 영향을 준다.',
    '',
    'Crop vs 전체 파이프라인 분해:']
for category,count in sorted(cross.items()):
    lines.append(f'{category}: {count}')
lines.append(f"Crop 오답 중 공백만 다른 사례: {len(whitespace_only)}개. 원인 분류용이며 공식 CER/EM은 공백을 제거하지 않고 계산했다.")
lines+=['','인식 crop 조건별 지표 (원인 확정이 아닌 관찰):','조건 | n | CER% | EM%']
for key,result in bucket_results.items():
    if result['n']:
        lines.append(f"{key} | {result['n']} | {result['cer_pct']:.4f} | {result['exact_match_pct']:.4f}")
lines+=['','수치 기반 우선순위',
    f"인식 crop에서는 맞았지만 전체 파이프라인에서 미매칭된 {cross['unmatched_gt_crop_correct']}줄과, 매칭 후 틀린 {cross['crop_exact_full_incorrect']}줄을 우선 살펴본다. 영역 병합·분할·잘림·회전과 입력 차이의 영향을 구분할 수 있는 표본이다.",
    f"공백만 다른 {len(whitespace_only)}개는 전사·학습 라벨의 공백 정책을 먼저 검토한다. 점수를 높이기 위해 평가 규칙에서 공백을 제거하지 않는다.",
    '25자 초과 440줄과 사전 밖 문자 포함 85줄은 별도로 추적한다. 길이가 긴 경우 CER과 EM을 함께 보고, 첫 학습에서 사전/헤드를 임의 변경하지 않는다.',
    '', '향후 진행 순서',
    '1. 현재 진단 결과와 실패 예시를 기준으로 인식 오류와 검출·영역 분할 문제를 구분한다. crop이 맞고 full이 틀리는 사례를 인식 학습만으로 해결된다고 가정하지 않는다.',
    '2. 동일 실제 영수증·재촬영을 묶은 그룹 정의를 만든다. receipt_id와 동일 파일 SHA만으로 재촬영 독립성을 보장하지 않는다. 최종 일반화 평가용 신규/미사용 영수증 그룹을 따로 확보한다.',
    '3. verified train 그룹만으로 사전·구조 유지의 작은 LR 미세조정을 설계한다. OOV·긴 문자열을 보고하고 학습 제외와 평가 포함을 구분한다.',
    '4. 고정된 동일 crop/검출 캐시로 전후 비교하고 KORIE validation 회귀도 확인한다. 모든 정책을 val에서 고정한 뒤 새 최종 test를 연다.',
    '',
    '증거 경로',
    'crops_diagnostic_v1/predictions.jsonl, summary.json, by_receipt.json: 줄별 및 영수증별 인식 결과.',
    'aligned_partial_v1/partial_predictions.jsonl, summary.json: 누락을 포함한 부분 GT 평가.',
    'aligned_partial_v1/<receipt>_<hash>/detection_cache.json, *.png, preview.jpg: 재사용 가능한 검출 crop, 모든 예측, 시각화. 주황 박스는 미매칭 GT.',
    'raw_preview_v2/: 사전 규칙으로 선택한 20장 원본 PNG의 정성 OCR. aligned GT를 원본 좌표에 억지로 대응시키지 않았다. raw_preview_v1은 좌표 자료형 오류로 중단된 시도를 보존한 것이다.',
    'analysis_v2/analysis.json: 교차 분류, 조건별 지표, 실패 예시. 금액 필드 GT가 없어 금액 정확도는 산출하지 않았다.',
    'execution_plan_v1.json 및 각 protocol.json: 고정된 모델·입력·정책. 동일 런타임 mogumogu-korie-training:20261117.',
]
with (out/'평가결과_및_튜닝제안_v1.txt').open('x',encoding='utf-8') as stream:
    stream.write('\n'.join(lines)+'\n')
write(base/'STATUS_completed_v2.json',dict(status='completed',scope='pre-tuning diagnostic only',crop=crop_summary,partial=full_summary,raw=raw_summary,report='analysis_v2/평가결과_및_튜닝제안_v1.txt'))
print('\n'.join(lines[:15]))
