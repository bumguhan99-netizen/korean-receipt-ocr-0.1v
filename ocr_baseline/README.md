# KORIE PP-OCRv5 인식 모델 비교

GLM-OCR 추가 모델 비교와 실행 방법은 [GLM_OCR.md](GLM_OCR.md)에 정리합니다.

루트 `AGENTS.md`에 따라 구현한 Recognition-only CPU 평가입니다. 데이터는 `../dataset/korie`의 Crop과 동일 stem TXT를 읽습니다. 개인 영수증 52장은 이 실험에 포함하지 않습니다.

## 실행

이 폴더에서 PowerShell로 실행합니다. 현재 `.venv`에는 Python 3.11.9, PaddlePaddle 3.2.0 CPU, PaddleOCR 3.7.0, PaddleX 3.7.2가 설치되어 있습니다. 전체 의존성 버전은 `requirements.lock.txt`에 고정했습니다.

```powershell
# 새 환경을 만드는 경우
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

# 환경 기록, 지표 검사, 원본 데이터 검사
.venv\Scripts\python.exe scripts\env_check.py
.venv\Scripts\python.exe scripts\check_metrics.py
.venv\Scripts\python.exe scripts\audit_dataset.py --make-clean-split --seed 42

# 두 공식 모델 다운로드 및 커밋/파일 해시 기록
.venv\Scripts\python.exe scripts\download_models.py --model-root models

# 같은 seed로 선택한 Val 100개, 이후 Val 전체
.venv\Scripts\python.exe scripts\evaluate.py --split val --device cpu --max-samples 100 --seed 42 --require-local-models
.venv\Scripts\python.exe scripts\evaluate.py --split val --device cpu --require-local-models

# 저장 결과의 지표 재계산 및 모델 간 입력 일치 검증
.venv\Scripts\python.exe scripts\verify_results.py outputs\baseline\official\val\smoke_100_seed42
.venv\Scripts\python.exe scripts\verify_results.py outputs\baseline\official\val\full
.venv\Scripts\python.exe scripts\write_report.py
```

이미 존재하는 실행 폴더는 덮어쓰지 않습니다. 재실행이 필요하면 `--run-tag cpu_retry_01`처럼 별도 이름을 사용하고 이유를 기록합니다. 실패 시 `run_info.json`은 `incomplete`로 기록되며 `error.log`가 생성됩니다. 실패 샘플을 건너뛰어 점수를 내지 않습니다.

모델 다운로드를 재실행하면 기존 `models/*/download_manifest.json`에 기록된 커밋을 유지하고 해시를 검사합니다. 최신 upstream 버전으로 바꾸는 작업은 `download_models.py --refresh`로 명시해야 하며, 기존 Baseline과 별도 실험으로 기록해야 합니다.

`--data-root`로 다른 데이터 위치를 명시할 수 있습니다. 기본 경로는 지침의 절대 경로입니다. 원본 이미지와 TXT를 이동하거나 수정하지 않습니다. 이미지 해시와 정답을 추론 직전 다시 검사하여 오래된 Manifest 사용을 차단합니다.

## 평가 조건

- `TextRecognition` 두 인스턴스로 이미 잘린 텍스트 이미지의 문자열만 인식합니다.
- CPU, batch 1, CPU threads 4, MKL-DNN 비활성화를 두 모델에 동일하게 적용합니다. 가중치·문자 사전은 각 공식 모델의 고유 구성입니다.
- 두 모델의 공식 `PreProcess`가 동일하며 CTC 디코딩을 사용하는지 검사합니다. 추가 이미지 처리나 결과 문자열 보정은 하지 않습니다.
- GT는 파일 끝 CR/LF만 제거합니다. 비교 시 Unicode NFC만 적용하며 공백·대소문자·쉼표는 보존합니다.
- CER/WER는 편집거리의 micro 합산 비율입니다. 숫자·기호 위주 Exact Match는 가격뿐 아니라 날짜·식별번호 등을 포함할 수 있습니다.

공식 분할에는 동일 이미지와 영수증 ID의 누출이 있습니다. `outputs/data_audit.json`에서 실제 수치를 확인하세요. 사전학습 모델의 공식 Val 비교 결과를 공식 Train 파인튜닝 후의 일반화 성능 근거로 사용하면 안 됩니다.

`clean_manifest.csv`는 동일 이미지 해시를 중복 제거하고 영수증 그룹 단위 seed 42로 새로 나눈 별도 실험입니다. 서로 다른 영수증 ID가 같은 이미지 해시를 공유하는 경우 그룹도 연결합니다. 공식 KORIE 분할과 직접 비교할 수 없습니다. clean 평가는 `--manifest outputs/clean_manifest.csv`로 별도 실행합니다.

## 결과와 Test

환경·감사 결과는 `outputs/`, 모델별 예측·필드별 지표·오류 50개는 `outputs/baseline/official/val/`에 저장합니다. CSV는 UTF-8 BOM입니다. `run_info.json`에 실행 상태, 버전, 장치, 입력 Manifest 해시, 모델 커밋·가중치 해시와 샘플 수를 기록합니다.

이번 단계에서는 Test 추론을 실행하지 않습니다. 최종 Test는 Val에서 조건을 확정한 후 한 번만 실행합니다. 실행 코드는 `--frozen-test-config` JSON의 `manifest_sha256`, `settings`, `model_downloads`가 현재 조건과 정확히 맞고 로컬 고정 모델을 사용할 때만 Test를 허용합니다. 이 세 값은 확정한 Val 실행의 `run_info.json`에서 가져올 수 있습니다. Test 실행 시작 표식을 독점 생성하므로 다른 `--run-tag`로 재평가하는 것도 차단합니다. Test의 파일·중복 검사는 성능 평가에 해당하지 않습니다.

인터페이스 근거: [PaddleOCR TextRecognition 공식 문서](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html), [범용 모델](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec), [한국어 모델](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec). 공개 README의 성능 수치와 본 실험이 동일한 조건이라고 가정하지 않습니다.
