import json
from pathlib import Path
from common import MODELS, read_csv, write_csv
from verify_glm_results import verify


def main():
    root=Path('outputs')
    smoke=root/'glm/official/val/smoke_100_seed42'
    full=root/'glm/official/val/full'
    verify(smoke)
    verify(full)
    info=json.loads((full/'run_info.json').read_text(encoding='utf-8'))
    comparisons=read_csv(full/'comparison_summary.csv')
    write_csv(root/'glm_comparison.csv',comparisons)
    smoke_predictions=read_csv(smoke/'predictions.csv')
    full_predictions=read_csv(full/'predictions.csv')
    by_path={r['image_path']:r for r in full_predictions}
    consistency=sum(r['prediction']==by_path[r['image_path']]['prediction'] for r in smoke_predictions)
    lines=['# KORIE GLM-OCR 추가 비교 결과','',
        '공식 GLM-OCR 가중치를 로컬에서 실행한 결과입니다. 기존 두 PaddleOCR 모델과 같은 공식 Val Crop·정답·평가 지표를 사용했습니다. Val 100개 스모크 테스트 후 전체 Val을 측정했습니다. Test 추론은 수행하지 않았습니다.','',
        '## 전체 Val 결과','',
        '| 모델 | 장치 | n | CER % ↓ | WER % ↓ | Exact Match % ↑ | 숫자·기호 n | 숫자·기호 Exact % ↑ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['model']} | {r['device']} | {r['n']} | {float(r['cer_pct']):.2f} | {float(r['wer_pct']):.2f} | {float(r['exact_match_pct']):.2f} | {r['numeric_symbol_only_n']} | {float(r['numeric_symbol_only_exact_pct']):.2f} |")
    lines+=['',
        '평가 이미지 바이트·순서·정답·NFC 정규화와 지표는 일치합니다. GLM은 GPU BF16, 기존 PaddleOCR는 CPU이며 모델 고유 전처리와 디코딩 방식도 다릅니다. 이 표는 같은 데이터의 인식 결과 비교이며 동일 하드웨어의 처리 속도 비교가 아닙니다.','',
        '## GLM 스모크 테스트','',
        '| n | CER % | WER % | Exact Match % |','|---:|---:|---:|---:|']
    for r in read_csv(smoke/'summary.csv'):
        lines.append(f"| {r['n']} | {float(r['cer_pct']):.2f} | {float(r['wer_pct']):.2f} | {float(r['exact_match_pct']):.2f} |")
    lines+=['',f'스모크 100개 중 전체 실행에서도 동일한 예측을 낸 샘플은 {consistency}개입니다. 입력 목록과 GT는 모두 일치합니다.','',
        '## 필드별 전체 Val 결과','',
        '| 필드 | n | 범용 CER % | 한국어 CER % | GLM CER % | 범용 Exact % | 한국어 Exact % | GLM Exact % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    paddle=Path(info['reference_run'])
    fields=[{r['field']:r for r in read_csv(paddle/name/'by_field.csv')} for name in MODELS]
    fields.append({r['field']:r for r in read_csv(full/'by_field.csv')})
    assert set(fields[0])==set(fields[1])==set(fields[2])
    for field in sorted(fields[0]):
        a,b,c=[f[field] for f in fields]
        assert a['n']==b['n']==c['n']
        scores=[float(r[k]) for k in ('cer_pct','exact_match_pct') for r in (a,b,c)]
        lines.append(f"| {field} | {a['n']} | "+' | '.join(f'{v:.2f}' for v in scores)+' |')
    lines+=['','## 생성 조건과 재현성','',
        f"- 모델: `{info['model']}`; HF 커밋 `{info['model_download']['revision']}`.",
        f"- 모델 가중치 SHA-256: `{info['model_download']['files']['model.safetensors']}`.",
        f"- 데이터 Manifest SHA-256: `{info['manifest_sha256']}`.",
        f"- 패키지: `{info['packages']}`.",
        f"- 장치: `{info['settings']['device']}`, `{info.get('gpu','')}`; 정밀도 `{info['settings']['dtype']}`.",
        '- 공식 `Text Recognition:` 프롬프트 고정, greedy, batch 1, max_new_tokens 512, SDPA. Detection·레이아웃 분석·IE를 사용하지 않았습니다.',
        '- 생성된 응답 토큰만 디코딩하고 특수 토큰을 제외했습니다. strip·공백 제거·줄바꿈 변환·가격 보정·별도 LLM 후처리는 하지 않았습니다. 응답에 포함된 서식과 줄바꿈도 원문 그대로 채점합니다.',
        f"- 생성 길이 상한 도달: {info['token_limit_reached']}개. 상한에 닿은 결과도 제외하지 않고 채점했습니다.",
        '- GLM 생성 결과에는 PaddleOCR와 비교 가능한 calibrated recognition confidence를 제공하지 않으므로 confidence는 공란입니다.',
        '- 모든 샘플의 이미지 해시·GT와 모델 파일 해시를 재확인했습니다. 저장된 예측에서 전체·필드별 지표를 다시 계산하여 검증했습니다.',
        '- 공식 분할의 중복/영수증 ID 누출은 기존 data_audit.json에 기록된 그대로입니다. 공식 Train 파인튜닝 후의 독립 일반화 성능을 뜻하지 않습니다.',
        f"- GLM 전체 Val 추론 시간: {info['inference_seconds']:.1f}초. 모델 로딩 시간은 별도이며 기존 CPU 모델과 속도를 직접 비교하지 않습니다.",'',
        '## 결과 파일','',
        '- [세 모델 요약 CSV](glm_comparison.csv)',
        '- [GLM 전체 예측 CSV](glm/official/val/full/predictions.csv) · [필드별 CSV](glm/official/val/full/by_field.csv) · [오류 상위 50개](glm/official/val/full/worst_50.csv)',
        '- [실행 정보](glm/official/val/full/run_info.json) · [검증 결과](glm/official/val/full/verification.json) · [환경](glm_environment.json)',
        '- [실행 방법](../GLM_OCR.md)', '',
        '공식 출처: [GLM-OCR 모델](https://huggingface.co/zai-org/GLM-OCR), [Transformers GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr).','']
    (root/'glm_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote outputs/glm_report.md and outputs/glm_comparison.csv')


if __name__=='__main__':
    main()
