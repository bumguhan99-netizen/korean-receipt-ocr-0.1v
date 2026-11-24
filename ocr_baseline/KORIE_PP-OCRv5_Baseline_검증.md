# KORIE 한국어 영수증 OCR: PP-OCRv5 두 모델 Baseline 검증 지침

> 작성 기준: 2026-11-17 · Windows 10/11 · PaddleOCR 3.x · 학습 전 Recognition-only 비교
>
> **이 문서를 코딩 에이전트(Codex/Claude Code)에 그대로 전달하여 구현·실행을 맡길 수 있다.** 동봉된 `scripts/`는 바로 실행 가능한 출발점이다. 실행 환경·데이터 위치는 사용자 PC에서 확인해야 하며, 확인되지 않은 GPU나 로컬 파일 구조를 추정하지 말 것.

## 0. 이번 단계의 목적과 범위

- 목표: **`PP-OCRv5_mobile_rec`(범용) vs `korean_PP-OCRv5_mobile_rec`(한국어 전용)** 를 **동일한 KORIE 이미지 Crop**에 추론하여 **CER, WER, Exact Match**, 숫자·기호 위주 문자열 정확도와 필드별 결과를 CSV로 저장한다.
- 작업 단위: `TextRecognition` **단독**. **Detection, IE, YOLO, Full OCR Pipeline은 사용하지 않는다.** 전처리·디코딩·후처리는 두 모델에서 동일 조건으로 유지한다.
- 주요 검증 집합: **Val**. 처음에는 100개 Crop으로 스모크 테스트, 그다음 Val 전체. **Test는 사전에 평가/정규화 조건을 고정한 후 단 한 번 최종 보고**. Train은 나중의 파인튜닝용이며, 현 단계에서는 파일·중복 검사 용도.
- 하드웨어: 사용자 Windows PC 사양은 아직 확인하지 않았다. 우선 **CPU**에서 동작 확인하고, NVIDIA GPU·드라이버·Paddle 호환성 확인 후 GPU 실행은 별도 검증한다.
- **실제 OCR 성능 수치는 아직 측정하지 않았다.** 아래의 데이터 개수 및 중복 개수는 제공된 공개 ZIP을 파일 단위로 검사해 얻은 값이다.

## 1. 경로(절대 변경하거나 자동 이동하지 말 것)

```text
DATA_ROOT = C:\Users\SSAFY\Desktop\mogu\mogumogu\dataset\korie
PROJECT_ROOT = C:\Users\SSAFY\Desktop\mogu\mogumogu\ocr_baseline
```

원래 데이터셋을 이동하거나 덮어쓰지 않는다. 제안하는 폴더 구조:

```text
mogumogu/
├─ dataset/
│  └─ korie/
│     ├─ train/          # 실제 구조는 재귀 스캔하여 확인
│     ├─ val/
│     └─ test/
└─ ocr_baseline/
   ├─ KORIE_PP-OCRv5_Baseline_검증.md
   ├─ requirements.txt
   ├─ .gitignore
   ├─ scripts/
   │  ├─ env_check.py
   │  ├─ audit_dataset.py
   │  ├─ download_models.py
   │  ├─ metrics.py
   │  ├─ check_metrics.py
   │  └─ evaluate.py
   ├─ .venv/              # 설치 후 생성
   ├─ models/             # 모델 다운로드 후 생성
   └─ outputs/            # 검사·평가 후 생성
```

**중요:** 이 문서와 동봉된 스크립트를 `ocr_baseline/` 폴더에 넣어 실행한다. 다른 위치에 배치할 경우 PowerShell에서 해당 폴더로 이동하고 `--data-root`에 정확한 절대경로를 전달한다.

## 2. 공개 KORIE ZIP에서 확인한 실제 형식

공개 출처:

- KORIE 저장소: <https://github.com/MahmoudSalah/KORIE>
- [Train ZIP](https://drive.google.com/file/d/1I4BzOqKgF7zbNPlNeood4f7g8pi2xh26/view)
- [Val ZIP](https://drive.google.com/file/d/1v_0iGpBjB5WdWOeKI4C903eeqkRBjTsM/view)
- [Test ZIP](https://drive.google.com/file/d/1GtEzSUA2wTNfOujO67-JEZ_PLpJOdBhg/view)

2026-11-17 기준, 다운로드된 세 ZIP의 **실제 파일 내용**:

| Split | JPG 이미지 | TXT 라벨 | 이미지-라벨 쌍 누락 | TXT 빈 라벨 |
|---|---:|---:|---:|---:|
| train | 5,356 | 5,356 | 0 | 0 |
| val | 1,785 | 1,785 | 0 | 0 |
| test | 1,786 | 1,786 | 0 | 0 |
| **합계** | **8,927** | **8,927** | **0** | **0** |

대표 파일 형태:

```text
val/IMG00001_MerchantName.jpg
val/IMG00001_MerchantName.txt   # UTF-8 텍스트: 포시애를-청주
val/IMG00001_Total.jpg
val/IMG00001_Total.txt          # UTF-8 텍스트: 38,000
```

- 이미지 확장자는 검사한 ZIP에서 모두 `.jpg`였다. 라벨은 동일한 상대경로·파일명 stem의 `.txt` 파일이다.
- 라벨 TXT는 해당 Crop의 **정답 문자열 1개**이며, 검토된 공개 ZIP에는 멀티라인·빈 라벨이 없다.
- 파일명 `IMG00001_...`의 앞부분은 영수증 그룹 ID, 뒤는 필드 이름(예: `MerchantName`, `Item_Total_Price`, `Total`)으로 사용한다.
- 파일명이 필드명 기반이므로 **영수증 전체 이미지를 바로 OCR 평가한다고 오해하지 말 것**. 이는 이미 잘라진 Crop을 인식하는 **Recognition 평가**이다.
- **README의 17,587 OCR crops와 현재 3개 ZIP의 8,927 이미지 수는 일치하지 않는다.** 데이터 버전 또는 통계 정의가 다른 것인지 확인되지 않았다. 논문·보고서에는 ZIP 실제 해시, 파일 수와 README 수치를 구분해 기록할 것.
- 이 수치는 제공된 원격 ZIP을 실제 확인한 내용이다. 사용자 PC의 `train/val/test` 폴더가 동일한지는 **반드시 로컬 감사 스크립트로 재확인**한다.

### 2.1 중요: 공식 분할에 데이터 누출이 있다

세 ZIP의 이미지 바이트 SHA-256과 파일명 영수증 ID를 직접 비교한 결과:

| 비교 | 동일 영수증 ID 개수 | 동일 이미지 SHA-256 개수 |
|---|---:|---:|
| Train ↔ Val | 641 | **155** |
| Train ↔ Test | 638 | **149** |
| Val ↔ Test | 606 | **52** |

전체 8,927 Crop에 대해 **서로 다른 이미지 해시 8,291개, 서로 다른 영수증 ID 674개**가 확인되었다. 동일 이미지 해시의 정답 문자열 간 충돌은 발견되지 않았다.

**해석:** 이미 사전학습된 두 모델의 단순 Val 비교에는 사용할 수 있지만, **공식 Train으로 파인튜닝한 뒤 공식 Val/Test 개선을 엄밀한 일반화 근거로 주장하면 안 된다.** 공식 Test에도 동일 이미지와 같은 영수증에서 나온 다른 Crop이 존재한다.

연구용 추가 실험은 다음처럼 분리한다.

1. **`official` 실험:** 출처에서 제공한 공식 Train/Val/Test를 보존해 참고 결과 기록. 다른 논문의 공식 수치와 직접 비교하는 경우에도 평가 구현 차이와 누출을 명시.
2. **`clean` 실험:** 세 공식 분할을 합치고 동일 이미지 해시를 중복 제거한 후, `IMGxxxxx` **영수증 ID 단위** 80/10/10 재분할(시드 42). 영수증 ID·이미지 해시가 분할 사이에 겹치면 즉시 중단. `clean`은 **공식 Test와 별개의 새 실험 분할**이고 공식 KORIE 수치와 직접 비교 불가.
3. 직접 스마트폰으로 촬영한 영수증 약 60장은 **추후 외부 일반화 평가**에만 활용. 개인정보 마스킹, GT 라벨 작성 전에는 현재 실험에 혼합하지 않는다.

## 3. Windows 실행환경 확인 및 설치

**PowerShell** 기준. 기존 시스템 Python·Git 환경을 덮어쓰지 않는다. `py -0p`로 먼저 설치된 Python을 확인하고, 예제는 **Python 3.11 x64가 있는 경우** 실행한다. 없으면 Python 3.11 x64를 설치하거나, 현재 Python이 Paddle/PaddleOCR 지원 범위인지 확인한 후 명령을 조정한다.

```powershell
# (1) Windows PowerShell
$DATA = 'C:\Users\SSAFY\Desktop\mogu\mogumogu\dataset\korie'
$PROJECT = 'C:\Users\SSAFY\Desktop\mogu\mogumogu\ocr_baseline'
cd $PROJECT

# (2) PC 및 Python 확인
py -0p
python --version
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
nvidia-smi  # NVIDIA가 없다면 명령을 찾을 수 없어도 됨

# (3) 독립 Python 환경: PowerShell Activate.ps1 실행 정책을 바꿀 필요 없음
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# (4) PaddlePaddle CPU 버전 (PaddleOCR 공식 설치 문서의 안정 버전 예시)
.\.venv\Scripts\python.exe -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# (5) PaddleOCR + 다운로드 의존성
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# (6) 동작 및 환경 기록
.\.venv\Scripts\python.exe -c "import paddle; print('Paddle:', paddle.__version__); print('CUDA:', paddle.is_compiled_with_cuda())"
.\.venv\Scripts\python.exe scripts\env_check.py
.\.venv\Scripts\python.exe scripts\check_metrics.py
```

- 패키지 버전은 재현성을 위해 `paddlepaddle==3.2.0`, `paddleocr==3.7.0`으로 시작한다. 설치 호환성 문제가 있으면 실제 오류를 확인한 뒤 **버전을 함께 기록하여** 조정한다. 모델은 명시적으로 PP-OCRv5를 선택하므로 PaddleOCR의 최신 기본 OCR 모델이 바뀌어도 혼동하지 않는다.
- **CPU 실험을 먼저 끝낸 뒤** GPU로 전환한다. GPU 사용 시 Windows용 Paddle GPU 빌드 지원 여부·NVIDIA 드라이버·필요 CUDA 조합을 공식 문서에 따라 확인할 것. `nvidia-smi` 출력만 보고 임의의 GPU wheel 설치 금지.
- 관련 문서: [PaddleOCR 설치](https://www.paddleocr.ai/main/en/version3.x/installation.html), [PaddlePaddle 설치](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html).

### 3.1 실제 로컬 데이터 폴더를 먼저 살펴보기

```powershell
Test-Path $DATA
Get-ChildItem $DATA -Directory | Select-Object FullName
Get-ChildItem (Join-Path $DATA 'val') -Recurse -File | Select-Object -First 15 FullName
```

`train`, `val`, `test` 중 하나라도 없다면 평가로 넘어가지 않는다. ZIP 내부에 최상단 `val/`처럼 폴더가 중첩되어 있어도 스크립트는 해당 분할 디렉터리 하위를 재귀 검색한다.

## 4. 데이터 감사 및 라벨 매핑

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --data-root "$DATA" --output-root outputs
```

필수 검사 내용:

- `train/val/test` 폴더 존재 및 각 이미지 개수.
- 이미지마다 같은 stem의 `.txt` 라벨 존재, 반대로 고아 TXT 없음, UTF-8 읽기 가능, 빈 라벨 없음.
- 라벨의 실제 내용을 사람이 확인할 샘플 출력 또는 CSV 제공. 정답 문자열에 대하여 `.rstrip('\r\n')`로 **텍스트 파일 끝 줄바꿈만 제거**하고, 앞뒤 일반 공백·구두점·쉼표는 보존.
- 이미지 SHA-256 및 영수증 ID `IMG\d+`로 중복 검출.
- 해시가 동일한데 GT가 다르면 결과를 무효 처리하고 수동 조사.
- 실제 개수·중복 검사를 `outputs/data_audit.json`에 저장.
- 평가 데이터 목록을 `outputs/official_manifest.csv`에 UTF-8 BOM CSV로 생성.
- **검사 실패 시 파일 누락을 조용히 무시하거나 점수를 계산하지 않는다.**

Manifest 스키마:

| 열 이름 | 의미 |
|---|---|
| `eval_split` | 현재 평가 분할: `train` / `val` / `test` |
| `source_split` | 원본 ZIP 위치에 따른 분할 |
| `receipt_id` | `IMG00001` 등 영수증 그룹 키 |
| `field` | `MerchantName`, `Total`, `Item_Unit_Price` 등 |
| `image_path` | DATA_ROOT 기준 상대 경로 (예: `val/IMG00001_Total.jpg`) |
| `gt_text` | 원본 TXT에서 읽은 정답 문자열 |
| `image_sha256` | 동일 이미지 파일 검출용 SHA-256 |

후속 연구를 위한 **별도** 영수증 단위 재분할을 만들려면:

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --data-root "$DATA" --output-root outputs --make-clean-split --seed 42
```

이는 `outputs/clean_manifest.csv`와 `outputs/clean_split_info.json`을 추가로 만들며 원본 이미지·라벨은 **어떠한 경우에도 수정하지 않는다**. 재분할은 처음부터 새로운 데이터셋 실험으로 관리한다.

## 5. 두 모델 다운로드

공식 모델:

- 범용: <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec>
- 한국어: <https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec>
- `TextRecognition` 공식 문서: <https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html>

동봉된 스크립트는 Hugging Face의 모델 저장소 커밋 SHA를 확인해 해당 스냅샷을 `models/<model_name>/`로 다운로드하고, `models/download_manifest.json`에 **가중치 SHA-256과 저장소 커밋**을 기록한다.

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --model-root models
```

다운로드 후 적어도 다음 파일이 있는지 확인:

```text
models/PP-OCRv5_mobile_rec/inference.json
models/PP-OCRv5_mobile_rec/inference.pdiparams
models/PP-OCRv5_mobile_rec/inference.yml
models/korean_PP-OCRv5_mobile_rec/inference.json
models/korean_PP-OCRv5_mobile_rec/inference.pdiparams
models/korean_PP-OCRv5_mobile_rec/inference.yml
```

> 다운로드 실패 시 재시도하기 전에 프록시, 디스크 용량, 인증/방화벽 등을 점검. **대안:** PaddleOCR가 모델 이름으로 가중치를 자동 다운로드하도록 `--require-local-models` 옵션을 생략하고 실행할 수 있다. Hugging Face 접근이 어려운 환경에서는 공식 문서의 PowerShell 환경변수 `$env:PADDLE_PDX_MODEL_SOURCE='BOS'` 를 설정해 모델 소스를 변경할 수 있다. 이 경우 실제 모델 로딩 위치와 버전을 따로 기록할 것.

## 6. 실제 모델 추론 인터페이스(수정하면 안 되는 핵심)

모델 2개를 별도로 초기화하고 **이미지 Crop 경로를 직접 전달**한다. 전체 영수증에 `PaddleOCR()` 파이프라인을 돌리는 코드는 이번 Recognition 실험과 다르므로 사용하지 않는다.

```python
from paddleocr import TextRecognition

# 둘 다 같은 설정으로 실행한다. CPU부터 확인.
model = TextRecognition(
    model_name="korean_PP-OCRv5_mobile_rec",
    model_dir="models/korean_PP-OCRv5_mobile_rec",  # 명시적으로 다운로드했을 때
    device="cpu",
)
results = model.predict(input="path/to/single_crop.jpg", batch_size=1)

for result in results:
    payload = result.json  # 공식 결과의 JSON 표현
    # 예시 형태: {"res": {"rec_text": "38,000", "rec_score": 0.9, ...}}
    data = payload.get("res", payload)
    prediction = data["rec_text"]
    confidence = data.get("rec_score")
```

위 예시는 **형식 설명용**이다. 실제 실행은 동봉된 `evaluate.py`를 이용한다. `result.json` 접근이 버전에 따라 다르게 동작한다면 JSON 저장 메서드로 파싱하되, `rec_text`가 확인되지 않으면 오류로 중단하고 임의 문자열로 대체하지 않는다.

**모델 비교 공정성 규칙**:

1. 동일한 Manifest·동일한 이미지 Crop 순서·동일한 평가 샘플.
2. 동일한 `batch_size=1`, 동일한 `device=cpu`, 동일한 PaddleOCR/PaddlePaddle 버전.
3. 입력 크기·이미지 파일 바이트 변경 금지. 첫 실험에서는 추가 확대·샤프닝·그레이스케일·기하 보정·TTA 금지.
4. 프롬프트, 사전, Regex 가격 보정, LLM, 숫자 복원 등의 후처리 금지.
5. 모델별 결과를 각각 저장하고, **추론 실패 샘플을 몰래 제외하지 않기**. 에러 발생 시 해당 실행을 미완료로 처리.
6. CPU로 확립한 Baseline을 GPU에서 반복하면 별도 실행 이력으로 남기기.

## 7. 평가 지표(정의 고정)

정규화는 **Unicode NFC만** 적용한다. 원래 GT TXT의 파일 끝 개행은 미리 제거하지만, 평가 문자열 자체에서는 **공백 제거, 소문자 변환, 숫자 쉼표 삭제, 특수문자 삭제, 한국어 형태소 분리 등을 하지 않는다**.

- **CER (micro, %)** = `100 × sum(Levenshtein(GT 문자열, 예측 문자열)) / sum(GT 문자 수)`
- **WER (micro, %)** = `100 × sum(Levenshtein(GT.split(), Pred.split())) / sum(GT.split() 토큰 수)`
- **Exact Match (%)** = `100 × NFC 정규화 후 문자열 완전 일치 Crop 수 / 전체 Crop 수`
- **숫자·기호 위주 Exact Match (%)** = GT에 숫자가 1개 이상 있고 영문/한글이 없는 Crop 집합에서 Exact Match를 계산. 가격뿐 아니라 날짜·전화번호·식별번호도 섞일 수 있으므로 **가격 정확도라고 부르면 안 된다**.
- **필드별 CER/WER/Exact Match** = `Total`, `MerchantName`, `TransactionDate`, `Item_Unit_Price` 등의 `field`별로 별도 집계. 표본수 `n`도 함께 저장.

주의사항:

- 문자열 1개가 통째로 GT이므로 빈칸 또는 문자가 1개라도 틀리면 Exact Match는 실패.
- 한국어 WER은 **공백 기준 토큰**으로 계산한다. KORIE의 Crop이 단어 1개인 경우가 많아 WER이 OCR 전반의 단어 분할 성능을 의미하지 않는다. 정밀 비교는 CER과 Exact Match를 함께 본다.
- **전체 데이터셋의 CER는 Crop별 CER 평균이 아니라 미시(micro) 합산**으로 산출한다. WER도 동일.
- 모든 숫자는 `%`로 저장한다. CER/WER은 삽입 오류 때문에 100% 초과도 가능하다.
- 검증 스크립트 `scripts/check_metrics.py`를 먼저 실행하여 쉼표(`,`)나 공백을 지우는 정규화가 의도치 않게 들어가 있지 않은지 점검한다.

## 8. 실행 명령 — Val 100개 → 전체 Val → 최종 Test

먼저 **두 모델이 같은 이미지 100개를 실제로 추론하는지** 확인한다. 스모크 샘플은 고정 시드 42로 동일하게 고른다.

```powershell
# 조건 1: 사전 다운로드·데이터 감사 완료
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --data-root "$DATA" `
  --manifest outputs\official_manifest.csv `
  --split val `
  --device cpu `
  --max-samples 100 `
  --seed 42 `
  --require-local-models
```

그다음 **Val 전체**를 실행한다. `--max-samples`는 빼야 한다.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --data-root "$DATA" `
  --manifest outputs\official_manifest.csv `
  --split val `
  --device cpu `
  --require-local-models
```

**Test는 모델·전처리·평가지표 및 변경 사항을 Val에서 확정한 뒤에만 실행한다.**

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --data-root "$DATA" `
  --manifest outputs\official_manifest.csv `
  --split test `
  --device cpu `
  --require-local-models
```

추후 **영수증 단위 clean 재분할**의 Val 평가:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --data-root "$DATA" `
  --manifest outputs\clean_manifest.csv `
  --split val `
  --device cpu `
  --require-local-models
```

이후 Train을 이용한 파인튜닝을 구현하더라도 **모델 선정·데이터 증강은 clean Val만**, 최종 평가는 clean Test만 사용한다. 공식 분할 결과와 clean 분할 결과는 같은 표에 섞어 직접 비교하지 않는다.

## 9. 자동 저장 결과

예상 파일 구조:

```text
outputs/
├─ environment.json
├─ data_audit.json
├─ official_manifest.csv
├─ clean_manifest.csv                  # --make-clean-split 선택 시
├─ clean_split_info.json               # --make-clean-split 선택 시
└─ baseline/
   └─ official/
      └─ val/
         ├─ smoke_100_seed42/
         │  ├─ summary.csv
         │  ├─ run_info.json
         │  ├─ PP-OCRv5_mobile_rec/
         │  │  ├─ predictions.csv
         │  │  ├─ metrics.json
         │  │  ├─ by_field.csv
         │  │  └─ worst_50.csv
         │  └─ korean_PP-OCRv5_mobile_rec/...
         └─ full/
            ├─ summary.csv
            ├─ run_info.json
            ├─ PP-OCRv5_mobile_rec/...
            └─ korean_PP-OCRv5_mobile_rec/...
```

`predictions.csv` 필수 열:

```text
model,source_split,eval_split,receipt_id,field,image_path,image_sha256,
ground_truth,prediction,confidence,
ground_truth_norm,prediction_norm,
char_edits,gt_chars,word_edits,gt_words,exact_match,numeric_symbol_only
```

`summary.csv`는 최소 다음을 포함:

```text
model,n,cer_pct,wer_pct,exact_match_pct,
numeric_symbol_only_n,numeric_symbol_only_exact_pct,
char_edits,gt_chars,word_edits,gt_words,
eval_split,manifest_name,limited_to_n,device,normalization,tokenization
```

`run_info.json`에 반드시 기록할 내용: Python 및 PaddleOCR/PaddlePaddle 버전, 실행 장치, 모델 HF 커밋 및 weight SHA-256(다운로드 Manifest), 데이터 Manifest 해시, 분할, 샘플 수, seed, batch 크기, 정규화 방식. 모델별 추론 완료 여부도 확인.

## 10. 검증 완료 기준(Acceptance Criteria)

- [ ] `py -0p`, CPU/GPU 탐지 및 `outputs/environment.json`에 실행환경 저장.
- [ ] 로컬 `DATA_ROOT`의 세 분할을 재귀 탐색, 이미지/텍스트 수와 모든 대응 관계 검증.
- [ ] 해시 중복·영수증 ID 누출을 `data_audit.json`에 보고; CSV Manifest 손실 없이 생성.
- [ ] 두 모델을 **서로 다른 실제 가중치**로 다운로드하고 HF 커밋·weight 해시 기록.
- [ ] 단일 Crop을 넣어 `rec_text`와 `rec_score` 정상 추출 확인.
- [ ] 동일한 Val 100개에 두 모델을 비교하고 `predictions.csv`를 각각 생성.
- [ ] 두 모델의 sample ID 집합/개수/GT가 완전히 동일한지 확인.
- [ ] CER/WER/Exact 계산의 단위 테스트 통과. 엑셀에서 한글이 정상 보이도록 UTF-8 BOM 저장.
- [ ] Val 전체에서 CER/WER/Exact, 필드별 결과, 숫자·기호 위주 Exact, 오류 상위 50개 저장.
- [ ] 추론·입력·출력 오류 발생 시 임의로 샘플을 건너뛰지 않음.
- [ ] 공식 Test는 최종 조건 고정 전 실행하지 않음.
- [ ] 정량 성능 값은 실제 모델 실행 후에만 기입하며 README 수치와 직접 동일 조건인 것처럼 비교하지 않음.

## 11. 코딩 에이전트에게 전달할 명확한 구현 요청

> 이 `KORIE_PP-OCRv5_Baseline_검증.md` 문서에 따라 Windows PC의 `C:\Users\SSAFY\Desktop\mogu\mogumogu\dataset\korie` 에 있는 **실제** train, val, test를 검사하고, `PP-OCRv5_mobile_rec`과 `korean_PP-OCRv5_mobile_rec` 인식 모델의 공정한 Baseline 비교 코드를 구현하라. **데이터/하드웨어를 임의로 가정하지 말 것.** 먼저 각 split의 이미지·정답 TXT의 실제 구조를 검사하여 사람이 확인할 리포트를 생성하고, 누락·중복·train/val/test 누출을 보고하라. CPU 우선, 두 모델의 가중치를 동일한 PaddleOCR 환경에서 다운로드 및 버전 고정하고 Val 100개→Val 전체 순서로 추론하라. 정규화는 Unicode NFC만, CER·WER는 micro Levenshtein, Exact Match와 필드별 지표를 계산하라. 원본 데이터는 수정하지 말고 UTF-8 BOM CSV와 JSON으로 모든 예측·GT·점수·실험 환경을 저장하라. Test는 최종 설정 확정 전 실행하지 말 것. **데이터 누출이 발견된 공식 분할에서 파인튜닝 일반화 성능을 과장하지 말고, 영수증 ID 단위 clean 재분할도 별도 지원하라.** 정상 실행 여부를 실제 PC의 출력과 오류 로그로 입증하라.

## 12. 원문 출처 및 주의

- [KORIE GitHub README](https://github.com/MahmoudSalah/KORIE): Crop 데이터 및 공개 Baseline 설명. README의 PaddleOCR CER 15.84%는 **모델 정확한 버전·정규화·평가 설정이 같다고 검증되지 않았으므로** 본 결과와 동등 비교 금지.
- [PaddleOCR Text Recognition](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html): `TextRecognition`, `predict`, `rec_text`, `rec_score`, `model_dir`, `device`.
- [PP-OCRv5 Korean Model Card](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec), [PP-OCRv5 General Model Card](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec): 모델 ID와 언어 범위.
- [PaddlePaddle 설치 문서](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html).
- 개인 영수증 이미지·거래 정보·연락처를 공개 Git 저장소에 업로드하지 말 것. 원본/학습 산출물 사용 조건은 KORIE의 공개 라이선스 및 배포 조건을 별도 확인할 것.
