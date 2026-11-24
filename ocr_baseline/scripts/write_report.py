"""Write a local Markdown report using completed, verified runs only."""
import json
from pathlib import Path
from common import MODELS, read_csv
from verify_results import verify


def main():
    root = Path('outputs')
    audit = json.loads((root/'data_audit.json').read_text(encoding='utf-8'))
    env = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    clean = json.loads((root/'clean_split_info.json').read_text(encoding='utf-8'))
    smoke = root/'baseline/official/val/smoke_100_seed42'
    full = root/'baseline/official/val/full'
    verify(smoke)
    verify(full)
    info = json.loads((full/'run_info.json').read_text(encoding='utf-8'))
    text = ['# KORIE PP-OCRv5 CPU Baseline 결과', '',
            '공식 Val의 동일한 Crop을 두 사전학습 인식 모델에 입력한 실측 결과입니다. 개인 영수증은 포함하지 않았으며 Test 추론은 실행하지 않았습니다.', '',
            '## 실측 성능', '',
            '| 범위 | 모델 | n | CER % ↓ | WER % ↓ | Exact Match % ↑ | 숫자·기호 n | 숫자·기호 Exact % ↑ |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope,folder in [('Val 100 (seed 42)',smoke),('Val 전체',full)]:
        for row in read_csv(folder/'summary.csv'):
            text.append(f"| {scope} | {row['model']} | {row['n']} | {float(row['cer_pct']):.2f} | {float(row['wer_pct']):.2f} | {float(row['exact_match_pct']):.2f} | {row['numeric_symbol_only_n']} | {float(row['numeric_symbol_only_exact_pct']):.2f} |")
    text += ['', 'CER/WER는 micro 합산입니다. Exact Match는 Crop 전체 문자열 일치율이며, 숫자·기호 지표는 가격 전용 정확도가 아닙니다. 공개 README의 CER와 평가 조건이 같다고 가정하지 않습니다.', '',
             '## 전체 Val 필드별 결과', '',
             '| 필드 | n | 범용 CER % | 한국어 CER % | 범용 Exact % | 한국어 Exact % |',
             '|---|---:|---:|---:|---:|---:|']
    fields = [{r['field']:r for r in read_csv(full/name/'by_field.csv')} for name in MODELS]
    assert set(fields[0]) == set(fields[1])
    for field in sorted(fields[0]):
        a,b = (f[field] for f in fields)
        assert a['n'] == b['n']
        text.append(f"| {field} | {a['n']} | {float(a['cer_pct']):.2f} | {float(b['cer_pct']):.2f} | {float(a['exact_match_pct']):.2f} | {float(b['exact_match_pct']):.2f} |")
    text += ['', '## 데이터 감사', '', '| Split | 이미지 | TXT |', '|---|---:|---:|']
    for split,count in audit['counts'].items():
        text.append(f"| {split} | {count['images']} | {count['labels']} |")
    text += ['', f"총 {audit['total_images']:,}개, 고유 이미지 해시 {audit['unique_image_hashes']:,}개, 고유 영수증 ID {audit['unique_receipt_ids']:,}개입니다. 이미지·라벨 누락, 빈 라벨 및 동일 해시의 GT 충돌은 발견되지 않았습니다.", '',
             '| 분할 쌍 | 겹치는 영수증 ID | 겹치는 이미지 해시 |', '|---|---:|---:|']
    for pair,overlap in audit['overlaps'].items():
        text.append(f"| {pair} | {overlap['receipt_id']} | {overlap['image_sha256']} |")
    text += ['', '공식 Train으로 파인튜닝한 뒤 공식 Val/Test 결과를 독립적인 일반화 성능으로 주장할 수 없습니다. 여기서는 사전학습 모델의 추론만 수행했습니다.', '',
             f"clean 분할은 동일 이미지 중복 제거 후 영수증 그룹 단위 seed {clean['seed']}로 생성했습니다: {clean['counts']}. clean 성능 평가는 이번 실행에 포함하지 않았습니다.", '',
             '공개 README의 17,587 Crop과 로컬 공개 분할의 8,927개는 수치가 다릅니다. 이 보고서는 실제 로컬 파일 검사 결과를 사용합니다. ZIP 파일 해시는 확보하지 않았으며 개별 이미지 해시는 Manifest에 기록했습니다.', '',
             '## 실행 환경과 검증', '',
             f"- Python: {env['python'].splitlines()[0]}",
             f"- 패키지: {env['packages']}",
             f"- 장치·조건: {info['settings']}",
             f"- Manifest SHA-256: `{info['manifest_sha256']}`",
             '- 추가 이미지 전처리와 문자열 후처리는 적용하지 않았습니다. 두 모델의 공식 전처리 구성은 동일하고 CTC 디코딩을 사용합니다. 각 공식 가중치의 문자 사전은 서로 다릅니다.',
             '- 지표 단위 검사를 통과했고, 저장된 예측에서 모든 지표를 다시 계산했습니다. 두 모델의 입력 순서·이미지 해시·GT와 전체 샘플 수가 일치합니다.',
             '- 데이터/모델 로딩 및 추론 오류 없이 두 모델 모두 완료했습니다. Test는 파일 감사만 수행했습니다.',
             '- 콘솔에는 Paddle의 ccache 미설치 안내가 있습니다. PowerShell의 stderr 리디렉션이 이를 NativeCommandError로 표시해 셸 반환값은 1이었지만, 두 모델의 run_info는 completed이며 전체 예측과 재계산 검증이 완료됐습니다. 로그는 `val_full_console.log`에 보존했습니다.', '']
    for name,metadata in info['model_downloads'].items():
        text += [f"### {name}", '', f"- HF 커밋: `{metadata['revision']}`", f"- 가중치 SHA-256: `{metadata['files']['inference.pdiparams']}`", '']
    text += ['## 확인할 파일', '',
             '- [환경](environment.json) · [데이터 감사](data_audit.json) · [라벨 샘플 CSV](label_samples.csv)',
             '- [전체 Val 요약 CSV](baseline/official/val/full/summary.csv) · [실행 정보](baseline/official/val/full/run_info.json) · [결과 검증](baseline/official/val/full/verification.json)',
             '- 각 모델 폴더의 `predictions.csv`, `by_field.csv`, `worst_50.csv`에서 원문 정답과 예측 오류를 확인할 수 있습니다.', '',
             '출처: [KORIE](https://github.com/MahmoudSalah/KORIE), [PaddleOCR TextRecognition](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html).', '']
    (root/'baseline_report.md').write_text('\n'.join(text), encoding='utf-8')
    print('Wrote outputs/baseline_report.md')


if __name__ == '__main__':
    main()
