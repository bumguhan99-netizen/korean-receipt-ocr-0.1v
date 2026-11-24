"""Summarize only completed run artifacts, without changing selected checkpoints."""
import csv
import json
from pathlib import Path
import re
from datetime import datetime

root=Path(__file__).resolve().parent
run=root/'round2/run_001'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def examples(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    return len(rows),rows[:3]

pre=read(root/'evaluation/pre_round2_val/summary.json')
val=read(run/'validation/summary.json')
test=read(run/'final_test/summary.json')
selection=read(root/'final_selection_v1.json')
full=read(root/'evaluation/full_val_round2_v1/summary.json')
full_test=read(root/'evaluation/full_test_v1/summary.json')
for path,expected in [(run/'validation/run_info.json',1785),(run/'final_test/run_info.json',1786)]:
    info=read(path)
    if info['status']!='completed' or info['n']!=expected:
        raise ValueError('Incomplete evaluation')
if full['status']!='completed' or full_test['status']!='completed':
    raise ValueError('Incomplete full OCR')
log=(run/'outputs_train/train.log').read_text()
best=re.findall(r'best metric, acc: ([0-9.]+).*?best_epoch: (\d+)',log)
epochs=re.findall(r'epoch: \[(\d+)/30\]',log)
times=re.findall(r'\[(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)\]',log)
elapsed=(datetime.strptime(times[-1],'%Y/%m/%d %H:%M:%S')-datetime.strptime(times[0],'%Y/%m/%d %H:%M:%S')).total_seconds()
reserved=max(map(int,re.findall(r'max_mem_reserved: (\d+) MB',log)))
allocated=max(map(int,re.findall(r'max_mem_allocated: (\d+) MB',log)))
if not epochs or max(map(int,epochs))!=30:
    raise ValueError('Training did not complete 30 epochs')
improved,good=examples(run/'validation/improved.csv')
regressed,bad=examples(run/'validation/regressed.csv')
report=[
    'KORIE 전체 영수증 평가 및 2차 학습 최종 결과',
    '',
    f"최종 채택: {selection['selected']}",
    f"배포용 export: {selection['export']}",
    '선정: 내장 val accuracy 최고 epoch를 선택한 뒤, 전체 외부 val CER과 EM이 모두 1차보다 나쁘지 않을 때 2차를 채택했다. 최종 test는 선택 이후 실행했다.',
    '',
    '학습 및 복원',
    'Git LFS HEAD b1f72ad의 학습 가중치를 별도 폴더에 복원했다. 복원 export의 pdiparams SHA256이 기존 1차 선택 모델과 일치했다.',
    '공식 split·한국어 사전·구조 유지, seed42, LR1e-5, first_bs32, workers4, 최대30epoch. 같은 2차 실험의 optimizer만 사용하며 1차 optimizer는 가져오지 않았다.',
    f"본 학습 완료 epoch: {max(map(int,epochs))}; 내장 최고 val accuracy/epoch: {best[-1] if best else '로그 확인 필요'}",
    f'학습 로그 최초/최종 시각 간 실제 경과 {elapsed:.0f}초. 로그상 최대 GPU reserved {reserved}MB, allocated {allocated}MB. 전체 환경 설치 시간과 구분한다.',
    'smoke는 별도 64개 표본과 작은 배치의 2epoch였다. smoke 가중치를 본 학습에 사용하지 않았다.',
    '학습 중 주기별 내장 val 정확도의 변동이 컸다. 마지막 epoch를 채택하지 않고 최고 내장 val checkpoint를 export한 뒤 전체 외부 val/test에서 다시 측정했다. 변동 원인을 과적합으로 단정하지 않는다.',
    '',
    'Crop 평가 (NFC, 공백·구두점 유지; CER=총 편집거리/총 GT 문자)',
    'split | 모델 | n | CER% | EM% | 숫자기호 EM% | 금액필드 EM%',
]
for split,name,row in [('val','공식 baseline',pre['baseline']),('val','1차',val['baseline']),('val','2차',val['candidate']),('test','1차',test['baseline']),('test','2차',test['candidate'])]:
    report.append(f"{split} | {name} | {row['n']} | {row['cer_pct']:.4f} | {row['exact_match_pct']:.4f} | {row['numeric_symbol_only_exact_pct']:.4f} | {row['amount_exact_pct']:.4f}")
report += ['',f'Val 문자 편집거리 개선 샘플 {improved}개, 악화 샘플 {regressed}개. 동일 편집거리 변화는 별도일 수 있다.', '개선 예시: '+json.dumps(good,ensure_ascii=False),'악화 예시: '+json.dumps(bad,ensure_ascii=False),'',
    '전체 영수증',
    f"Val 원본 154장, 동일 검출 캐시 {full['boxes']}개 영역을 두 모델에 재사용. 텍스트가 달라진 영역 {full['changed_texts']}개. 이는 정확도 개선 수가 아니다.",
    f"Test 원본 {full_test['n']}장 처리, 실행 실패 {len(full_test['failures'])}건.",
    'Val round2 시각화: evaluation/full_val_round2_v1/*_round2.jpg. Test 시각화: evaluation/full_test_v1/<receipt>/baseline.jpg 및 candidate.jpg.',
    '검출 GT는 key-information 박스이며 문자열/클래스명 대응이 제공되지 않았다. 미매칭 예측을 모두 FP로 집계하지 않았고 전체 OCR CER/precision은 산출하지 않았다.',
    '정성 확인 사례 IMG00003: 합계 영역 분할, 상품 중량의 g/9 혼동 등 오류가 남았다. 이 test 관찰을 이용해 모델이나 정책을 다시 변경하지 않았다.',
    'OCR train과 detection val의 영수증 ID 중복은 135/154, test는 131/156이었다. ID 중복은 실제 좌표-전사 연결을 증명하지 않는다. 원본과 crop의 바이트 동일성 중복은 발견하지 못했다. 전체 영수증 수치를 독립 holdout 인식 성능으로 해석하지 않는다.',
    '',
    '재현 증거',
    'records/images_20261117_2150.json, records/training_packages_20261117_2150.txt, restored_identity_v1/identity.json, round2/run_001/train.yml 및 outputs_train/train.log.',
    'RTX 5070 Ti 16GB, 드라이버 616.56 (records/gpu_final_v1.csv). 전체 영수증 추론은 동일 eval Docker 이미지와 records/eval_packages_v1.json 버전을 유지했다.',
    'PaddleOCR commit b03f46425e8ff4442b268ce449e3eef758146cd4. Docker GPU 역전파/업데이트 통과; training 이미지 requirements 및 pip check 통과.',
    '가중치/평가 입력 해시는 각 run_info.json, frozen_test.json, frozen_full_test_v1.json에 저장했다.',
    '',
    '스마트폰 영수증 후속 작업',
    'receipt_labeled_평가설계_v1.txt 및 phone_receipt_eval_v1.py에 사전 평가 계획·코드를 작성했다. 실제 스마트폰 모델 추론/튜닝은 이번 추가 요청에서 실행하지 않았다.',
    '304장/verified 8343줄/excluded 1732줄. 완전한 검출 GT·단어 GT·금액필드 GT가 아니며 aligned 좌표계와 원본 촬영 좌표계를 구분한다.',
]
with (root/'최종_실험보고서_v1.txt').open('x',encoding='utf-8') as stream:
    stream.write('\n'.join(report)+'\n')
with (root/'STATUS_final_v1.json').open('x') as stream:
    json.dump(dict(status='completed',selection=selection,full_val=full,full_test=full_test,report='최종_실험보고서_v1.txt',phone_plan='receipt_labeled_평가설계_v1.txt'),stream,ensure_ascii=False,indent=2)
print('\n'.join(report[:8]))
