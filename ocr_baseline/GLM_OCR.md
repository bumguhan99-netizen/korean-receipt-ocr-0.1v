# GLM-OCR 추가 비교

사용자 요청에 따라 `zai-org/GLM-OCR`의 공식 가중치를 다운로드하여 KORIE Val에서 비교합니다. `AGENTS.md`의 기존 두 PaddleOCR 모델 실험을 확장한 별도 실행입니다. 인식 모델에 이미 잘린 Crop만 입력하며 Detection·레이아웃 분석·IE 파이프라인은 사용하지 않습니다.

원본 이미지·정답·NFC 정규화·CER/WER micro 합산·Exact Match·숫자/기호 Exact Match는 기존 실험과 같습니다. 기존 `outputs/baseline/official/val/*/samples.csv`를 직접 읽어 샘플과 순서를 고정합니다. 모델 고유의 전처리와 생성 방식은 PaddleOCR와 다르며 GLM은 공식 `Text Recognition:` 프롬프트를 고정 사용합니다. 출력에서 입력 프롬프트 토큰과 특수 토큰만 제외하고 공백, 줄바꿈, 구두점과 문자열을 그대로 채점합니다.

## 실행 환경

기존 PaddleOCR `.venv`와 분리된 `.venv-glm`을 사용합니다. PyTorch CUDA 12.8 wheel과 실제 RTX 5060 Ti의 BF16·SDPA 커널 실행을 확인합니다. CPU 단일 샘플 확인 후 GPU에서 Val 100개 → 전체 Val 순서로 측정합니다. 실행 장치와 정밀도가 기존 Paddle CPU 측정과 다르므로 처리 속도 비교에는 사용할 수 없습니다. GLM의 인식 confidence는 제공하지 않아 CSV에서 공란으로 기록합니다.

```powershell
# ocr_baseline 폴더에서 실행
py -3.11 -m venv .venv-glm
.venv-glm\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv-glm\Scripts\python.exe -m pip install -r requirements-glm.txt
.venv-glm\Scripts\python.exe scripts\download_glm.py
.venv-glm\Scripts\python.exe scripts\check_glm_env.py
.venv-glm\Scripts\python.exe scripts\check_metrics.py
.venv-glm\Scripts\python.exe scripts\evaluate_glm.py --device cpu --max-samples 1 --run-tag cpu_probe
.venv-glm\Scripts\python.exe scripts\evaluate_glm.py --device cuda:0 --max-samples 100
.venv-glm\Scripts\python.exe scripts\verify_glm_results.py outputs\glm\official\val\smoke_100_seed42
.venv-glm\Scripts\python.exe scripts\evaluate_glm.py --device cuda:0
.venv-glm\Scripts\python.exe scripts\verify_glm_results.py outputs\glm\official\val\full
.venv-glm\Scripts\python.exe scripts\write_glm_report.py
```

가중치는 `models/GLM-OCR/`에 저장하고 다운로드 커밋과 모든 모델 파일의 SHA-256을 `download_manifest.json`으로 고정합니다. 재다운로드 시 기존 커밋을 유지합니다. 평가 직전에 모델 파일 해시와 이미지/정답을 다시 검증합니다. `trust_remote_code=False`, `local_files_only=True`로 로컬에서 실행합니다.

실제 설치된 전체 버전은 `requirements-glm.lock.txt`에 기록했습니다. 기존 결과 폴더는 덮어쓰지 않습니다. 콘솔 로그가 필요하면 `scripts/run_logged.py --log outputs/새로그.log -- scripts/evaluate_glm.py ...`로 실행하면 stdout/stderr와 실제 종료 코드를 보존합니다.

생성 조건은 greedy, batch 1, `max_new_tokens=512`입니다. 길이 상한에 닿은 샘플은 제외하지 않고 `token_limit_reached`로 표시하고 그대로 채점합니다. 빈 예측도 오류로 제외하지 않고 채점합니다. 실행 오류가 나면 해당 실행은 `incomplete`이며 완성된 성능 결과로 사용하지 않습니다.

결과는 `outputs/glm/official/val/`에 기존 결과와 분리합니다. 각 실행에 예측 CSV, 전체/필드별 지표, 오류 50개, 샘플 목록, 모델 버전/장치/프롬프트/생성 조건과 완료 상태를 저장합니다. 완료된 100개·전체 실행을 검증하면 세 모델의 `comparison_summary.csv`도 생성합니다. Test 추론은 이번 비교에 포함하지 않습니다.

출처: [GLM-OCR 공식 모델](https://huggingface.co/zai-org/GLM-OCR), [Transformers GLM-OCR](https://huggingface.co/docs/transformers/model_doc/glm_ocr), [PyTorch 2.10.0 공식 설치 조합](https://pytorch.org/get-started/previous-versions/#v2100).
