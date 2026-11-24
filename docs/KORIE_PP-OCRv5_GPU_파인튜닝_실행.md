# KORIE 한국어 PP-OCRv5 GPU 파인튜닝 실행 가이드

> 파일 배치 안내: 이 문서는 docs/로 이동했습니다. 별도 지정이 없는 경로와 명령의 실행 기준은 프로젝트 루트이며, 학습 상태에 관한 기존 기록은 이번 파일 정리에서 재검증하지 않았습니다.

작성·조사일: 2026-11-17. 기준 지침: [AGENTS.md](../AGENTS.md).

> **후속 요청 반영:** 사용자가 코드 구현·실행 결과 및 PC에 맞는 설정을 요청하여 `training/`에 실행 코드를 추가했다. 아래의 최초 조사 당시 “미구현/미실행” 표시는 당시 상태이며, 현재 실행 방법과 변경 설정은 이 절을 우선한다. 원본 데이터와 기존 baseline 코드는 수정하지 않는다.

## 현재 구현과 PC용 설정

**최대 자원 활용 요청 반영:** `tune_resources.py`가 실제 train만으로 first_bs 32/64/128/256/512 및 workers 4/8/12를 시험한다. 유한한 loss, 정상 종료, GPU reserved memory 14,000 MiB 이하인 후보 중 실측 처리량이 가장 높은 설정을 `configs/selected_resources.yml`에 저장한다. 본 학습은 이 파일을 사용하고 매 epoch validation을 수행한다. 성능 시험의 checkpoint는 본 학습에 재사용하지 않는다. 로그·전력·클럭·VRAM·온도 표본은 `resource_probes/`에 보존한다. RAM/VRAM을 의미 없는 할당으로 채우지 않는다.

Windows 최고의 성능 모드 적용, GPU 최대 전력 180W 확인, 지원 최대 graphics 3090MHz/memory 14001MHz lock 명령 성공. 실제 동작 클럭은 부하·전력·온도에 따라 달라지므로 telemetry로 확인한다. 원복용 `training/restore_performance.ps1`은 관리자 실행 시 기존 균형 조정 모드와 자동 GPU 클럭을 복원한다.

- `training/korie.py prepare`: 공식 split 보존, 실제 이미지 decode, NFC 변환본, 사전/순서 대조, OOV·길이 보류 기록, 전체 raw GT manifest 생성.
- `training/korie.py config`: 공식 v3.7.0 한국어 YAML을 읽어 구조·사전을 유지한 완전한 YAML 생성. 기본 `--profile rtx5060ti`, 원래 공식 하이퍼파라미터는 `--profile official`로 선택.
- `training/korie.py evaluate`: 두 한국어 export를 같은 전체 crop에 평가, 기존 NFC CER/EM 및 숫자·금액 지표, 개선/악화 CSV 저장. OOV/길이 보류 샘플도 외부 평가에는 포함한다.
- `training/korie.py freeze`: 완료된 full val의 모델·데이터·환경을 동결. test는 동결 일치 및 한 번 실행 guard 적용.
- `training/gpu_check.py`, `training/test_korie.py`: GPU backward/update 검사 및 합성 데이터 테스트.
- `training/setup_wsl.sh`, `training/run_wsl.sh`: 독립 WSL 환경 준비, 검사→초기 baseline→smoke→하드웨어 확인→본 학습→export→val→동결 test 순서. 기존 출력이 있으면 덮어쓰지 않고 중단한다. 중단 후에는 로그를 확인하고 해당 단계부터 개별 명령으로 재개한다.

| 설정 | 공식 원본 | RTX 5060 Ti 16GB 시작값 |
|---|---:|---:|
| train first_bs / loader batch | 128 | 32 |
| 실제 multiscale batch (높이 32/48/64) | 128/85/64 | 32/21/16 |
| train workers | 8 | 4 |
| val workers / batch | 4 / 128 | 2 / 32 |
| 최대 epoch | 75 | 30 |
| learning rate | 0.0005 | 0.00001 |
| warmup epochs | 5 | 1 |

이는 최적값을 주장하는 것이 아니라 PC용 검증 시작값이다. 먼저 4/0/2(epoch) smoke 후 32/4/1(epoch) 메모리·loader 확인을 거쳐 본 학습한다. 본 학습은 smoke 가중치가 아닌 원래 pretrained부터 시작한다. 실제 사용 VRAM 및 속도에 따라 배치를 조정하면 변경사항을 기록한다. 공식 `fix_bs=false`와 scales를 유지하므로 32가 모든 step의 실제 batch라는 뜻은 아니다. 출처: [공식 sampler](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/data/multi_scale_sampler.py).

현재 자동 실행은 **공식 내장 validation acc의 best_accuracy**로 checkpoint를 선택한 뒤 전체 raw val을 외부 CER/EM으로 보고한다. 내부 val은 학습 가능 표본만 포함하므로 전체 raw 평가와 분모를 구분한다. 아래 최초 가이드의 외부 CER 최소 선택 제안과는 다른, 이번 실행에서 명시적으로 고정한 선택 규칙이다. test로 epoch를 선택하지 않는다.

WSL2에서 실행(설치/실행은 환경과 기존 출력 확인 후 한 번만):

```bash
cd /mnt/c/Users/SSAFY/Desktop/mogu/mogumogu
bash training/setup_wsl.sh
bash training/run_wsl.sh
```

`setup_wsl.sh`는 먼저 `training/bootstrap/uv-install.sh`가 준비되어 있어야 한다. 최초 준비 명령은 `mkdir -p training/bootstrap` 후 `curl -LsSf https://astral.sh/uv/install.sh -o training/bootstrap/uv-install.sh`이다. [uv 공식 설치](https://docs.astral.sh/uv/getting-started/installation/), [Python 설치](https://docs.astral.sh/uv/guides/install-python/). 시스템 Python 3.14를 교체하지 않고 프로젝트 안에 Python 3.11을 둔다. 실제 성공/실패는 `training/setup.log`, `training/records/`, `training/evaluation/`에서 확인한다.

현재 검출기/전체 영수증 정답 연결은 아직 확인되지 않았으므로 전체 영수증 파이프라인 테스트는 별도 구현·확인이 필요하다.

실제 [NRTRLabelEncode](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/data/imaug/label_ops.py)는 `max_text_length=25`일 때 24자 이상을 거부한다. 따라서 24~25자는 `nrtr_length`, 25자 초과는 `length`로 기록하여 내장 학습/평가 목록에서 보류하고 외부 전체 평가는 유지한다. `verify_encoder.py`가 생성된 목록 전체를 실제 MultiLabelEncode로 재검증하여 loader의 조용한 표본 대체를 예방한다.

목표는 `korean_PP-OCRv5_mobile_rec`의 recognition만 미세조정하여 한국어 영수증 인식을 개선하고 서버 배포 가능한 추론 모델을 만드는 것이다. 첫 실험은 공식 한국어 문자 사전과 모델 구조를 유지한 전체 파라미터 미세조정이다. 검출기는 학습하거나 교체하지 않는다.

**이번 작업은 조사와 이 문서 작성만 수행했다. 설치, 모델 다운로드, 데이터 변환, 학습, 새 성능 평가는 실행하지 않았다.** 아래 설치·학습 명령은 후속 작업용이다. 존재하는 코드라도 현재 Python 실행 문제가 해결되기 전에는 실행할 수 없다.

## 1. 확인한 파일 구조와 준비 상태

프로젝트 루트: `C:\Users\SSAFY\Desktop\mogu\mogumogu`.

```text
mogumogu/
  AGENTS.md
  dataset/
    korie/
      train/  # 5,356 JPG + 같은 stem TXT
      val/    # 1,785 JPG + 같은 stem TXT
      test/   # 1,786 JPG + 같은 stem TXT
    images/   # 개인 영수증 데이터: KORIE와 별개
    labels/
    splits/   # 개인 영수증 분할: KORIE에 사용하지 않음
    train.jsonl, validation.jsonl, test.jsonl
  scripts/
    prepare_dataset.py
    ocr_windows.ps1
  ocr_baseline/
    .venv/
    .venv-glm/
    README.md
    KORIE_PP-OCRv5_Baseline_검증.md
    requirements.txt, requirements.lock.txt
    scripts/
      common.py, metrics.py, check_metrics.py
      audit_dataset.py, evaluate.py, verify_results.py
      env_check.py, download_models.py, write_report.py
    models/
      PP-OCRv5_mobile_rec/
      korean_PP-OCRv5_mobile_rec/
        inference.json
        inference.pdiparams
        inference.yml
        download_manifest.json
      GLM-OCR/
    outputs/
      environment.json
      data_audit.json
      official_manifest.csv
      clean_manifest.csv, clean_split_info.json
      baseline_report.md
      baseline/official/val/full/
        summary.csv, run_info.json, verification.json
        korean_PP-OCRv5_mobile_rec/
          predictions.csv, metrics.json, by_field.csv, worst_50.csv
```

### 1.1 데이터 확인

이번 조사에서 실제 디렉터리 파일 수를 다시 세었으며 합계는 JPG 8,927개, TXT 8,927개다. `dataset/korie`에는 이 두 확장자만 확인되었다. 공식 분할은 별도 로컬 split TXT가 아니라 `train/val/test` 디렉터리로 표현된다. `validation`의 실제 폴더명은 **`val`**이다. `outputs/official_manifest.csv`는 기존 코드가 만든 파생 목록이지 공식 배포 파일은 아니다.

제공된 이미지는 crop이며, 동일 stem TXT가 해당 crop의 UTF-8 정답 문자열이다. 실제 확인 예:

| 이미지 | 라벨 파일 | 내용 |
|---|---|---|
| `dataset/korie/val/IMG00001_Total.jpg` | `IMG00001_Total.txt` | `38,000` |
| `dataset/korie/val/IMG00001_MerchantName.jpg` | `IMG00001_MerchantName.txt` | `포시애를-청주` |

기존 `audit_dataset.py`는 `(IMG\d+)_(.+)`로 `receipt_id`와 `field`를 얻는다. 이 ID는 그룹 연결 정보다. **원본 전체 영수증 이미지 경로·crop 좌표·원본 annotation과의 정확한 연결표는 로컬 KORIE 폴더에서 확인되지 않았다.** 개인 영수증의 JSON 좌표를 KORIE annotation으로 사용하지 않는다.

기존 감사 결과 `data_audit.json`은 누락·빈 라벨·해시별 정답 충돌이 없고 `status=passed`라고 기록한다. 이는 저장된 검사 결과이며 이번에 전체 이미지 decode를 다시 수행했다는 뜻은 아니다. 기존 코드는 `PIL.Image.verify()`를 사용하므로 후속 준비 검사에서는 실제 `load()`/OpenCV decode도 확인한다.

저장된 감사에서 분할 간 동일 이미지 SHA-256 수는 train↔val 155, train↔test 149, val↔test 52이고 영수증 ID 교집합도 기록되어 있다. **이 로컬 사본의 기존 검사 결과라는 범위로만 보고**하며 KORIE 전체 버전의 결함이나 확인하지 않은 라벨 오류를 단정하지 않는다. 이번 실험은 공식 분할을 유지하고 재분할·전체 수동 검수를 시작 조건으로 삼지 않는다. 기존 `clean_manifest.csv`는 사용하지 않는다. 공식 분할 결과를 완전히 독립적인 새 영수증 일반화 성능으로 확대 해석하지 않는다.

[KORIE 공식 README](https://github.com/MahmoudSalah/KORIE)는 OCR train/val/test 다운로드를 구분한다. README의 17,587 crops와 로컬 8,927개 차이의 원인·배포 ZIP 해시·정확한 데이터 버전은 **확인 필요**다. README에 CER/WER 수치는 있지만, 확인한 공개 내용만으로 정규화·분모·정확한 baseline 모델 버전까지 재현할 수 없다. 공식 평가 코드/논문의 세부 규칙은 **확인 필요**이며 아래 로컬 평가 규칙을 공식 규칙이라고 부르지 않는다.

### 1.2 기존 모델과 평가

한국어 모델의 로컬 파일은 **추론 모델**이다. `inference.pdiparams`를 학습용 `.pdparams`로 이름만 바꿔 쓰지 않는다. 현재 프로젝트에 PaddleOCR 학습 소스 checkout, 한국어 학습 YAML, 한국어 학습용 `.pdparams`, KORIE 학습 목록 변환기는 확인되지 않았다.

한국어 추론 모델 다운로드 기록:

- HF revision: `c02ecaf1f22bfd1c618cce154fd19185b47e663a`
- `inference.pdiparams` SHA-256: `cac3e5f12cf04aaa77f6a5bc704e4e736ef2908476551891d84b41b4e9090462`
- 출처: [로컬 manifest](../ocr_baseline/models/korean_PP-OCRv5_mobile_rec/download_manifest.json), [공식 모델](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec).

기존 공식 val 전체 결과(이번에 새로 측정한 값이 아님):

| 모델 | n | CER % ↓ | 문자열 EM % ↑ | 숫자·기호 n | 숫자·기호 EM % ↑ |
|---|---:|---:|---:|---:|---:|
| PP-OCRv5_mobile_rec | 1,785 | 44.2686 | 50.0280 | 1,465 | 60.1365 |
| korean_PP-OCRv5_mobile_rec | 1,785 | 30.8679 | 54.5658 | 1,465 | 59.6587 |

근거: [summary.csv](../ocr_baseline/outputs/baseline/official/val/full/summary.csv), [verification.json](../ocr_baseline/outputs/baseline/official/val/full/verification.json), [보고서](../ocr_baseline/outputs/baseline_report.md). 검증 JSON에는 1,785개 입력 일치·지표 재계산 통과가 저장되어 있다. test 성능은 기록하지 않는다.

`evaluate.py`는 `common.py`의 두 공식 모델 이름에 고정되어 있다. GPU 환경에서도 패키지 메타데이터를 `paddlepaddle`라는 CPU 배포명으로 조회하므로 GPU-only 설치에서는 실패할 수 있다. 임의 튜닝 모델 경로를 공식 모델 디렉터리에 덮어써 우회하지 않는다. **튜닝/export 모델의 공정한 비교 평가기 추가 구현이 필요하다.**

기존 KORIE baseline은 `TextRecognition` 단독으로 검출기를 쓰지 않는다. 개인 영수증 코드는 `Windows.Media.Ocr`를 사용한다. 따라서 유지할 **Paddle 검출 모델의 이름·버전·가중치 경로는 확인 필요**다. 없는 검출 모델을 임의의 PP-OCRv5 검출기로 정하지 않는다.

### 1.3 환경: 기록과 현재 상태 구분

| 항목 | 확인 결과 |
|---|---|
| 현재 조사 셸 | Windows PowerShell |
| GPU | 이번 `nvidia-smi` 조회: RTX 5060 Ti, 16,311 MiB, 드라이버 610.88 |
| 저장된 Python | `environment.json`, `.venv/pyvenv.cfg`: 3.11.9 x64 |
| 저장된 baseline 패키지 | PaddlePaddle CPU 3.2.0, PaddleOCR 3.7.0, PaddleX 3.7.2; dist-info 디렉터리도 존재 |
| 현재 Python 실행 | `py -0p`: `No installed Pythons found!`; `.venv/Scripts/python.exe`는 base Python을 못 찾음 |
| GPU Paddle | 해당 baseline에는 CPU 패키지 흔적만 확인. GPU 학습 환경 미검증 |
| CUDA Toolkit | `nvcc`가 현재 PATH에서 확인되지 않음. 설치되지 않았다고 단정할 수 없음 |
| WSL | `wsl.exe` 존재. 배포판 설치·WSL2 GPU 전달·Linux Python은 확인 필요 |
| 실제 학습 OS | 아직 미확정. 조사 OS와 학습 OS를 혼동하지 않음 |
| 프로젝트 commit | `git rev-parse HEAD` 실패: 현재 유효한 HEAD 미확인. staged/untracked 파일 존재; 변경하지 않음 |

따라서 예전의 “Paddle 미설치” 정보는 baseline 환경의 저장 상태와 다르지만, **현재 그 환경이 실행 가능하다고도 할 수 없다.** Python 경로 접근 문제인지 설치 변경인지 후속 확인한다. CUDA UMD 13.3은 로컬 CUDA Toolkit 버전이 아니다.

## 2. 환경 선택·설치 계획

### 2.1 권장 후보와 근거

**우선 후보: Linux x86_64 또는 GPU가 정상 전달되는 WSL2, Python 3.11, PaddlePaddle GPU 3.3.0 cu129, PaddleOCR 소스 v3.7.0.** Python patch 버전도 기록한다. 현재 Windows의 3.11.9 기록을 WSL에도 설치되어 있다고 가정하지 않는다.

- [Paddle Linux 공식 pip 문서](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/linux-pip_en.md)는 Python 3.11과 `paddlepaddle-gpu==3.3.0`의 cu129 설치를 제시한다. CUDA/cuDNN 등의 런타임은 wheel 설치 경로로 제공되며 드라이버는 별도다.
- [RTX 50 공식 환경 가이드](https://www.paddleocr.ai/v3.3.2/en/version3.x/pipeline_usage/PaddleOCR-VL-RTX50.html)는 CUDA 12.9 이상을 지원하는 드라이버와 cu129 Paddle 3.2.1 이상을 안내한다. 이는 VL 환경 안내이므로 **PP-OCRv5 recognition 역전파 성공의 증거로 사용하지 않는다**. VL 전용 safetensors·vLLM 패키지는 이 작업에 설치하지 않는다.
- [Windows 일반 설치 문서](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/windows-pip_en.md)는 GPU 3.2.2 cu129를 안내하지만, [PaddleOCR 설치 문서의 Windows 50-series 절](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html#3-install-paddlepaddle-wheel-packages-for-nvidia-50-series-gpus-on-windows)은 전용 wheel과 **문자 인식 학습의 알려진 문제**를 명시한다. Windows 일반 wheel 안내만으로 이 모델의 학습 지원을 확정하지 않는다.
- [v3.7.0 한국어 YAML](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml)의 존재를 확인했다. baseline의 PaddleOCR 버전과 맞춰 조사 범위를 고정한다. 학습은 pip 추론 패키지만 설치하는 것이 아니라 저장소의 `tools/train.py`와 학습 의존성을 사용한다. [학습·export 설치 안내](https://www.paddleocr.ai/main/en/version3.x/installation.html#2-install-training-and-export-dependencies)

이 조합은 **공식 자료에 근거한 검증 시작 후보**이며 본 장비에서 검증된 완성 조합은 아니다. 문서들 간 버전 차이를 숨기지 않는다. 아래 기본 연산과 실제 OCR 짧은 학습이 모두 통과해야 확정한다.

### 2.2 즉시 가능한 읽기 전용 확인 — Windows PowerShell

실행 위치: 프로젝트 루트. 입력: 로컬 환경/기존 파일. 출력: 콘솔만. 성공 기준: GPU 확인, Python 경로·버전 확인, WSL 사용 시 VERSION=2 확인. 명령 실패도 기록한다.

```powershell
Set-Location 'C:\Users\SSAFY\Desktop\mogu\mogumogu'
py -0p
nvidia-smi
Get-Command python,python3,nvcc,wsl -ErrorAction SilentlyContinue
wsl --list --verbose
Get-Content ocr_baseline\.venv\pyvenv.cfg
Get-Content ocr_baseline\outputs\environment.json
```

Python 실행 경로가 해결되지 않았으면 설치·학습 명령으로 넘어가지 않는다. 기존 baseline venv를 GPU로 덮어쓰지 않는다.

### 2.3 Linux/WSL2용 후속 설치 명령

다음 `training/`은 **새로 생성할 작업 디렉터리**이며 현재 존재하는 것으로 확인한 경로가 아니다. WSL 기본 C 드라이브 mount가 유효한 경우만 아래 경로를 사용한다. Linux 별도 서버에서는 `PROJECT`를 실제로 확인한 프로젝트 위치로 바꾼다. Windows venv를 Linux에서 재사용하지 않는다.

실행 위치: WSL2/Linux Bash. 입력: 기존 데이터, 공식 저장소/패키지. 출력: `training/.venv`, `training/PaddleOCR`, `training/records`. 성공 기준: 경로 검증, pip check 통과, checkout commit과 버전 기록. **이번 작업에서는 실행하지 않은 설치 명령이다.**

```bash
export PROJECT='/mnt/c/Users/SSAFY/Desktop/mogu/mogumogu'
test -d "$PROJECT/dataset/korie/train" || exit 1
test -d "$PROJECT/dataset/korie/val" || exit 1
test -d "$PROJECT/dataset/korie/test" || exit 1
uname -a
nvidia-smi
python3.11 --version
python3.11 -c "import platform; print(platform.architecture(), platform.machine())"
mkdir -p "$PROJECT/training/records"
cd "$PROJECT/training"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
cd PaddleOCR
python -m pip install -r requirements.txt
python -m pip check
git rev-parse HEAD > ../records/paddleocr_commit.txt
git status --short > ../records/paddleocr_status.txt
python -m pip freeze > ../records/training_packages.txt
python --version > ../records/python.txt
nvidia-smi > ../records/nvidia-smi.txt
```

`python3.11`/venv 모듈이 없으면 해당 배포판에서 설치할 방법과 권한은 **확인 필요**다. 시스템 CUDA Toolkit을 무조건 설치/교체하지 않는다. Linux 파일 시스템으로 작업 위치를 옮기는 경우 실제 경로·원본 데이터 해시를 기록하고 위 변수를 다시 확인한다.

### 2.4 Windows 네이티브는 선택적 호환성 조사 경로

Linux/WSL2 경로와 별개 환경이다. Python 3.11 x64를 실제 사용할 수 있을 때만 실행한다. 실행 위치: 프로젝트 루트. 출력: 새 `training-win/.venv`, pip 콘솔. 공식 Windows RTX 50 전용 cp311 wheel URL은 아래와 같다. 이 개발 wheel이 현재 PaddleOCR v3.7.0과 함께 학습에 성공하는지는 **확인 필요**이며 본 학습 권장 조합으로 확정하지 않는다.

```powershell
py -3.11 -m venv training-win\.venv
$TrainPython = Join-Path $PWD 'training-win\.venv\Scripts\python.exe'
& $TrainPython -m pip install --upgrade pip
& $TrainPython -m pip install 'https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp311-cp311-win_amd64.whl'
& $TrainPython -m pip check
& $TrainPython -c "import paddle; print(paddle.__version__, paddle.version.cuda()); print(paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

성공 기준은 설치 완료가 아니라 다음 역전파 검사와 OCR 시험 학습까지다. Windows에서 실패하면 기존 CPU baseline을 유지하고 Linux/WSL2 후보를 검증한다. 서로 다른 Paddle CPU/GPU wheel을 같은 venv에 중복 설치하지 않는다.

### 2.5 GPU 연산·역전파 확인

실행 위치: `training/PaddleOCR`, 활성화된 학습 venv, Bash. 입력: 임의 작은 tensor. 출력: 콘솔. 성공 기준: CUDA 빌드 true, GPU ≥1, GPU 장치에서 유한한 loss/gradient와 실제 parameter 갱신, `run_check()` 통과.

```bash
python - <<'PY'
import paddle
print('Paddle', paddle.__version__, 'build CUDA', paddle.version.cuda())
assert paddle.is_compiled_with_cuda()
assert paddle.device.cuda.device_count() >= 1
paddle.set_device('gpu:0')
print(paddle.device.get_device())
paddle.seed(42)
net = paddle.nn.Linear(8, 4)
opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=net.parameters())
before = net.weight.detach().clone()
x = paddle.randn([4, 8])
loss = (net(x) ** 2).mean()
loss.backward()
assert bool(paddle.isfinite(loss).item())
assert net.weight.grad is not None
assert bool(paddle.isfinite(net.weight.grad).all().item())
opt.step()
assert bool((net.weight != before).any().item())
opt.clear_grad()
print('backward/update OK', float(loss.item()))
paddle.utils.run_check()
PY
```

Windows에서는 같은 Python 본문을 PowerShell `@' ... '@ | & $TrainPython -`로 전달할 수 있다. 위 연산이 성공해도 convolution/CTC/NRTR 학습 kernel을 모두 검증한 것이 아니므로 6절의 실제 OCR 시험 학습을 생략하지 않는다.

## 3. 공식 crop을 그대로 사용하는 데이터 준비

### 3.1 기존 검사 코드 사용

실행 위치: 프로젝트 루트, PowerShell. 입력: `dataset/korie`; 출력: 새 `training/records/data_check_v1/`. 전제: 기존 CPU Python 환경 복구. 성공 기준: 감사 `passed`, 5,356/1,785/1,786, 원본 경로 유지. 이 코드는 이미지/라벨 기본 검사에 더해 해시 교집합도 기록하지만 재분할 옵션은 사용하지 않는다.

```powershell
.\ocr_baseline\.venv\Scripts\python.exe ocr_baseline\scripts\audit_dataset.py --data-root dataset\korie --output-root training\records\data_check_v1
.\ocr_baseline\.venv\Scripts\python.exe ocr_baseline\scripts\check_metrics.py
```

Linux 학습 환경 준비 후에는 프로젝트 루트에서 같은 기존 스크립트를 아래처럼 실행할 수 있다. 출력 디렉터리가 이미 있으면 새로운 이름으로 바꾸어 기존 증거를 보존한다.

```bash
cd "$PROJECT"
training/.venv/bin/python ocr_baseline/scripts/audit_dataset.py --data-root dataset/korie --output-root training/records/data_check_v1
training/.venv/bin/python ocr_baseline/scripts/check_metrics.py
```

### 3.2 추가 구현 필요: 검사·학습 목록 변환기

현재 기존 스크립트는 CSV manifest까지만 만들며 PaddleOCR용 목록, 사전 OOV, 길이·비율 분포 검사는 없다. **다음은 구현 명세이며 존재하지 않는 명령을 제시하지 않는다.**

입력:

- `dataset/korie/{train,val,test}`의 원본 JPG/TXT 또는 기존 official manifest와 원본 대조 결과.
- 4절의 정확한 사전, 최종 max_text_length, 정규화 정책, 데이터 root.

출력(모두 새로 생성할 경로):

- `training/data/{train,val,test}_list.txt`: UTF-8, 한 줄당 `상대이미지경로<TAB>정답`.
- `training/data/manifest.jsonl`: `source_split`, `eval_split`, `receipt_id`, `field`, `image_path`, `label_path`, 이미지/라벨 SHA-256, `gt_raw`, `gt_train`, 크기, 변환 규칙.
- `training/data/validation_report.json`, `issues.csv`: 분할별 입력/출력/보류 수, 누락·decode·파싱·OOV·길이 초과 사유와 샘플 ID. 여러 사유 중복 집계와 고유 샘플 수를 구분.
- `training/data/train_smoke_list.txt`: 공식 train에서만 seed 42로 고른 최대 64개 정상 표본. 원본 train 전체 목록은 보존. smoke 선택 ID도 기록.

형식 예(아래 `<TAB>`은 설명용이며 실제 파일에는 탭 한 글자):

```text
val/IMG00001_Total.jpg<TAB>38,000
val/IMG00001_MerchantName.jpg<TAB>포시애를-청주
```

`data_dir`는 `dataset/korie`로 설정한다. 파일명 공백은 허용하되 separator인 TAB/개행이 포함된 경로·라벨은 별도 보고하고 임의 치환하지 않는다. TXT 끝 CR/LF만 제거하고 `.strip()`으로 공백을 지우지 않는다. official split을 옮기거나 seed로 재분할하지 않는다. 제공 crop을 재절단·회전·재압축하지 않는다.

검사 순서: 파일 존재 → 실제 image decode → UTF-8/한 문자열 라벨 파싱 → 원본/학습 문자열 기록 → 사전 호환 → 길이/비율 분포 → 최종 label encoder 통과 여부. 미등록 문자를 무시하는 encoder 동작에 의존하지 않는다. 원본 라벨을 신뢰하며 전체 수동 검수는 요구하지 않는다.

분할마다 `원본 표본 수 = 사용 가능 고유 수 + 보류 고유 수`가 맞아야 한다. train에서 표현할 수 없는 표본은 원문을 보존한 채 사유와 비율을 기록하여 첫 실험의 학습 가능 부분집합을 명시한다. **val/test는 원래 전체 분모로 raw GT 기반 추론 평가한다.** 길이/OOV 때문에 내장 학습 평가에서 빠진 수를 별도 보고하고 그 부분집합 점수를 전체 성능이라고 하지 않는다.

### 3.3 crop이 없는 다른 배포본을 쓰게 되는 경우

이번 로컬 데이터에는 crop이 있으므로 필요 없다. 향후 crop 없는 배포본에서는 공식 원본 이미지·annotation 스키마·좌표계·텍스트 전사·split 연결을 먼저 확인한다. bbox면 범위 clipping, quadrilateral이면 순서가 확인된 좌표의 perspective crop을 사용하고 변환 행렬, bbox/polygon, 원본 receipt ID, split, annotation ID를 sidecar에 남긴다. 인식 모델 예측으로 GT를 대체하지 않는다. **공식 annotation 원본 경로·형식 및 이 crop 생성기는 확인/추가 구현 필요**다.

## 4. 문자 사전과 길이 정책

공식 v3.7.0 한국어 학습 YAML의 정확한 경로:

```text
configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml
ppocr/utils/dict/ppocrv5_korean_dict.txt
```

둘 다 **PaddleOCR checkout 기준**이다. 구버전 `korean_dict.txt`나 범용 사전을 대신 쓰지 않는다. 현재 로컬 추론 사전은 `ocr_baseline/models/korean_PP-OCRv5_mobile_rec/inference.yml`의 `PostProcess.character_dict`에 내장되어 있다. checkout 후 공식 사전과 이 목록의 문자·순서를 비교한다. `use_space_char=true`에 의해 추가되는 공백과 decoder가 관리하는 특수 토큰을 구분하며, CTC blank/SOS/EOS를 파일에 수동 추가하지 않는다. 학습/추론 사전 SHA-256과 비교 결과를 기록한다. 학습 가중치와 현재 HF 추론 가중치의 정확한 대응도 checkpoint 로딩 및 학습 전 예측으로 검증한다.

공백, 숫자, 쉼표, 소수점, 마이너스, 대소문자를 보존한다. 기존 [metrics.py](../ocr_baseline/scripts/metrics.py)는 GT/예측에 **NFC만** 적용한다. 첫 변환본의 학습 문자열도 NFC로 생성하는 방안을 사용하되 raw GT와 NFC 변경 표본 수를 기록하고 원본 TXT는 수정하지 않는다. 사전 자체는 정규화하거나 정렬하지 않는다. val/test의 metric은 기존 코드 그대로 NFC를 양쪽에 적용한다. 이 정책을 KORIE 공식 평가 규칙으로 오인하지 않는다.

기본 max_text_length는 공식 YAML에서 25다. 최종 값은 **확인 필요**:

1. train과 val의 raw/NFC 문자 길이 p50/p95/p99/max, 25 초과 수, 이미지 가로세로비와 높이 48로 환산한 폭 분포를 집계한다. test는 설정 선택에 사용하지 않는다.
2. OOV는 문자별 코드포인트·빈도·샘플별 위치까지 기록한다. 사전 확장은 이번 첫 실험 범위 밖이다. OOV를 지워 정답을 바꾸지 않는다.
3. 우선 25/기본 폭으로 시험하고 초과를 숨기지 않는다. 길이 증가가 필요하면 NRTR positional parameter 등 checkpoint shape 호환성을 확인한다. 가중치 재초기화가 필요하면 기존 구조 유지 실험과 분리한다.
4. `Global.max_text_length`만 CLI override하면 YAML anchor로 이미 해석된 NRTR/증강 길이가 자동 갱신된다고 가정하지 않는다. 전체 YAML에서 anchor와 참조를 일관되게 수정한다.
5. 폭 조정은 train sampler scales, RecConAug image_shape, Eval RecResizeImg, d2s/export 및 추론 전처리를 함께 점검한다. CTC time step 수가 라벨 길이 및 반복 문자에 충분한지도 확인한다. 폭 변경 실험에서는 baseline도 같은 전처리로 재평가한다.

[공식 label encoder 소스](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/data/imaug/label_ops.py)의 길이 제한·미등록 문자 처리와 실제 encode 결과를 검사할 것. 내장 evaluator는 label encode를 거치므로 raw 문자열 전체를 보존한 외부 평가기를 대체하지 못한다.

## 5. 학습용 가중치·YAML 준비

### 5.1 공식 학습 가중치 확보 — 후속 실행

실행 위치: `training/PaddleOCR`, Bash, 학습 venv 활성화. 입력: 공식 pretrained URL. 출력: 새 `training/pretrained/`와 YAML 사본. 성공 기준: 파일 존재/해시 기록 및 시험 학습에서 구조 불일치 없이 로드. 아래 다운로드는 이번에 실행하지 않았다.

```bash
cd "$PROJECT/training/PaddleOCR"
mkdir -p ../pretrained ../configs ../outputs ../exports ../records
curl -fL 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams' -o ../pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams
cp configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml ../configs/korie_rec.yml
sha256sum ../pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams ppocr/utils/dict/ppocrv5_korean_dict.txt > ../records/initial_assets.sha256
```

URL 근거: [공식 Text Recognition 모델 목록](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md). 다운로드 재실행으로 파일을 덮어쓰지 말고 이미 있으면 해시와 출처부터 확인한다. 학습용 파일의 실제 내용/해시는 다운로드 후 **확인 필요**다.

### 5.2 YAML 변경 예시

**아래는 완전한 YAML이 아니라 공식 파일에 반영할 변경 부분이다. 이것만 저장하면 Architecture/Loss/transforms가 빠져 실행되지 않는다.** 나머지 공식 구조를 유지한 `training/configs/korie_rec.yml`을 작성해야 한다. 모든 상대경로는 `training/PaddleOCR`에서 실행한다는 기준이다.

```yaml
Global:
  use_gpu: true
  distributed: false
  seed: 42
  epoch_num: 20
  save_model_dir: ../outputs/korie_rec_v1
  save_epoch_step: 1
  eval_batch_step: [0, 100]
  pretrained_model: ../pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams
  checkpoints: null
  character_dict_path: ./ppocr/utils/dict/ppocrv5_korean_dict.txt
  use_space_char: true
Optimizer:
  lr:
    learning_rate: 0.00001
    warmup_epoch: 1
Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false
  is_filter: false
Train:
  dataset:
    data_dir: ../../dataset/korie
    label_file_list: [../data/train_list.txt]
  sampler:
    first_bs: 4
  loader:
    batch_size_per_card: 4
    num_workers: 0
Eval:
  dataset:
    data_dir: ../../dataset/korie
    label_file_list: [../data/val_list.txt]
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 4
    num_workers: 0
```

기본 구조는 PPLCNetV3 + SVTR_LCNet, MultiHead(CTC/NRTR), MultiLoss다. Backbone/head를 freeze하지 않는 전체 미세조정부터 시작한다. 작은 학습률 `1e-5`, 20 epochs, batch 4는 **검증된 최적값이 아닌 시작값**이다. 공식 Adam/Cosine/L2 설정을 유지하고 AMP는 첫 실험에서 추가하지 않는다. GPU 한 장은 `CUDA_VISIBLE_DEVICES=0`으로 제한한다.

MultiScaleSampler의 실제 batch는 scale에 따라 달라질 수 있으므로 loader batch만 바꾸지 말고 `first_bs`도 조정한다. batch 4로 시험 후 peak VRAM을 보고 8, 16 등으로 조정하되 실험별 기록을 남긴다. 16GB 전체를 사용 가능 메모리로 가정하지 않는다.

증강은 공식 Train.transforms의 RecConAug/RecAug만 train에서 사용한다. val/test에는 증강을 넣지 않는다. 새 영수증 증강이나 사전 확장은 이후 독립 실험이다. smoke 설정에서는 작은 데이터에서 반복이 충분하도록 batch와 epoch를 확인한다.

`RecMetric.acc`는 학습 중 상태 관찰용이며 `norm_edit_dis`는 micro CER가 아니다. [RecMetric 구현](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/metrics/rec_metric.py)과 로컬 NFC 평가가 다르므로 `best_accuracy`가 최종 배포 모델이라고 자동 확정하지 않는다. epoch별 checkpoint를 전체 val의 외부 평가기로 평가하여 **NFC micro CER 최소 → 동률이면 EM 최대 → 동률이면 이른 epoch** 순으로 고른다. 이 선택 규칙은 test 이전에 고정한다.

## 6. 실행 순서와 명령

아래는 단계별 전제가 충족된 뒤 실행한다. `training/` 이하 산출물은 **예정 경로**다. 존재하지 않는 사용자 스크립트는 실행 명령에 넣지 않았다. 공식 `tools/*.py`는 v3.7.0 소스에서 존재를 확인했지만 로컬에서는 clone 후 사용할 수 있다.

### 6.1 학습 전 baseline 확인·평가

지금은 저장된 `summary.csv`, `run_info.json`, `verification.json`을 읽을 수 있다. 기존 CPU 환경을 복구한 뒤 아래 명령으로 원래 지표를 재검증하거나 val을 다시 평가한다.

실행 위치: `ocr_baseline`, PowerShell. 입력: 공식 manifest와 기존 두 추론 모델. 출력: verify는 기존 결과의 검증 JSON, 재추론은 새 `outputs/baseline/official/val/pre_finetune_cpu_v1/`. 성공 기준: 1,785개, paired samples 일치, `status=completed`, 검증 passed. 기존 검증 JSON 보존이 필요하면 먼저 별도 보관한다.

```powershell
Set-Location 'C:\Users\SSAFY\Desktop\mogu\mogumogu\ocr_baseline'
.\.venv\Scripts\python.exe scripts\verify_results.py outputs\baseline\official\val\full
.\.venv\Scripts\python.exe scripts\evaluate.py --split val --device cpu --require-local-models --run-tag pre_finetune_cpu_v1
```

학습 시작 전 추가로 **동일 학습 환경에서 공식 pretrained와 튜닝 모델을 비교할 외부 평가기**를 준비한다. 기존 CPU 숫자와 GPU/폭 변경 숫자를 그대로 뺀 값을 순수 튜닝 효과로 제시하지 않는다. 공식 학습 가중치를 export한 초기 모델과 로컬 HF baseline의 val 예측도 비교해 모델 대응을 확인한다.

### 6.2 내장 학습 전 val 점검

실행 위치: `training/PaddleOCR`, Bash. 입력: 완성 YAML, 공식 pretrained, 변환된 val 목록. 출력: `training/records/native_baseline_val.log`. 성공 기준: 가중치 로드 성공, 유한한 metric, 예상 평가 수와 실제 수의 대조. **이 점수는 raw GT 전체의 최종 CER를 대체하지 않는다.**

```bash
cd "$PROJECT/training/PaddleOCR"
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
python tools/eval.py -c ../configs/korie_rec.yml -o Global.pretrained_model=../pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams 2>&1 | tee ../records/native_baseline_val.log
```

### 6.3 소량 데이터 실제 OCR 시험 학습

전제: 변환/사전/길이 검사 완료, 정상 train 표본 64개 목록과 val 목록 존재. 실행 위치: 동일. 입력: 원래 pretrained와 smoke train 목록. 출력: `training/outputs/korie_smoke/`, `records/smoke.log`. 성공 기준: 최소 여러 optimizer step, finite CTC/NRTR loss, backward 성공, eval 및 checkpoint 저장/재로딩 성공, shape mismatch 없음. GPU 인식만으로 성공 판정하지 않는다.

```bash
python tools/train.py -c ../configs/korie_rec.yml -o 'Train.dataset.label_file_list=[../data/train_smoke_list.txt]' Global.epoch_num=2 Global.eval_batch_step='[0,10]' Global.save_model_dir=../outputs/korie_smoke 2>&1 | tee ../records/smoke.log
```

loss 감소는 참고하되 이 작은 집합으로 일반화 성능을 판단하지 않는다. 이 테스트로 실패하면 본 학습을 시작하지 않는다. 본 학습은 smoke checkpoint를 이어받지 않고 원래 pretrained에서 시작한다.

### 6.4 본 학습과 validation 기반 선택

실행 위치: 동일. 입력: 공식 train/val의 준비된 목록, 원래 pretrained. 출력: `training/outputs/korie_rec_v1/`, `records/train_v1.log`. 성공 기준: loss/val 추이와 누락 수 기록, epoch checkpoint 생성, 데이터 loader의 무한 재시도/표본 대체 여부 확인.

```bash
python tools/train.py -c ../configs/korie_rec.yml 2>&1 | tee ../records/train_v1.log
```

내장 best checkpoint 점검(파일 생성 확인 후):

```bash
test -f ../outputs/korie_rec_v1/best_accuracy.pdparams || exit 1
python tools/eval.py -c ../configs/korie_rec.yml -o Global.checkpoints=../outputs/korie_rec_v1/best_accuracy 2>&1 | tee ../records/native_best_val.log
```

성공 기준: checkpoint 로딩 및 동일 val 내장 지표 확인. 최종 선정은 추가 외부 평가기로 epoch별 **전체 val**을 평가한 결과로 한다. 선택한 checkpoint의 실제 prefix를 `SELECTED`로 기록한다. 예측 성능을 이유로 test를 열어 epoch를 바꾸지 않는다.

### 6.5 추가 구현 필요: 공정한 모델 비교·최종 test·export 재평가

현재 `evaluate.py`의 `--model-root`는 두 고정 이름 하위 경로와 다운로드 manifest를 기대하므로 임의 모델 비교 옵션이 아니다. 다음 기능을 갖춘 별도 평가 진입점이 필요하다. **구현 전에는 튜닝/test 비교를 실행할 수 있는 것으로 표시하지 않는다.**

| 단계 | 실행 위치·입력 | 새 출력 경로 계획 | 성공 기준 |
|---|---|---|---|
| epoch val 평가 | 프로젝트 루트; 공식 manifest, epoch checkpoint 또는 각 export, 동일 사전/전처리 | `training/evaluation/val/<run_id>/` | 전체 1,785개 raw GT 평가, 예측/실패 수 일치, CER/EM 계산 |
| 최종 test 비교 | 프로젝트 루트; 확정 baseline·선정 튜닝 모델, test 목록, 동결 설정 | `training/evaluation/test/<run_id>/` | 동일 1,786개, 두 모델 한 쌍 비교, 설정/모델 해시 일치 |
| export 재평가 | 프로젝트 루트; 선택 모델의 export, 동일 val/test ID와 raw GT | `training/evaluation/export/<run_id>/` | export 전후 문자열 차이·CER/EM 차이 기록, 임의 샘플 누락 없음 |
| 개선/악화 비교 | 두 모델 predictions, image SHA/상대경로로 join | 각 run의 `improved.csv`, `regressed.csv` | GT·샘플 집합 동일, 문자 편집거리 증감과 필드·길이별 집계 |

평가기 구현 요구:

- baseline/finetuned를 구별하는 model ID 및 실제 model_dir/checkpoint를 입력받고 모델/사전/설정 해시 저장.
- `paddlepaddle` 또는 `paddlepaddle-gpu` 메타데이터를 실제 설치에 맞게 기록.
- 기존 `metrics.measure/aggregate`를 재사용해 NFC·micro 합산·숫자 집합 정의 유지.
- raw crop, native 전처리, batch 1, device/precision을 두 모델에 동일 적용. CPU 기존 조건은 MKL-DNN false, threads 4였다.
- val/test 라벨이 OOV/길이 초과라도 추론하고 원래 GT와 평가. 추론 실패는 실행 미완료로 기록하고 누락시킨 점수를 발표하지 않음.
- manifest/모델/정규화/전처리/금액 필드 whitelist를 담은 동결 JSON과 test 실행 표식을 구현. 재시도는 실패 이유와 동일 설정을 기록하며 모델 선택에 test를 사용하지 않음.
- 최종 test는 설정 고정 후 baseline/튜닝/export 검증을 미리 정한 동일 프로토콜로 수행. export 문제 발견 시 val로 돌아가 원인을 해결하고 test를 반복 선택 도구로 사용하지 않음.

기존 공식 모델 전용 test 명령은 `--frozen-test-config`가 필수이고 `manifest_sha256`, `settings`, `model_downloads`가 일치해야 한다. 오래된 baseline 지침의 동결 옵션 없는 test 예시는 현재 코드에서 동작하지 않는다. 기존 guard를 제거하거나 manifest를 위조하지 않는다.

### 6.6 학습 재개

실행 위치: `training/PaddleOCR`. 입력: 같은 실험의 실제 `latest` checkpoint prefix와 optimizer/state. 출력: 기존 실험 checkpoint 및 별도 resume log. 성공 기준: epoch·optimizer 상태 복원 기록. 파일명을 먼저 확인한다.

```bash
ls ../outputs/korie_rec_v1/
test -f ../outputs/korie_rec_v1/latest.pdparams || exit 1
test -f ../outputs/korie_rec_v1/latest.pdopt || exit 1
test -f ../outputs/korie_rec_v1/latest.states || exit 1
python tools/train.py -c ../configs/korie_rec.yml -o Global.checkpoints=../outputs/korie_rec_v1/latest 2>&1 | tee ../records/resume_v1.log
```

`Global.pretrained_model`은 초기 가중치 로딩이고 `Global.checkpoints`는 재개 용도다. optimizer/state가 없으면 완전한 resume라고 부르지 않는다. 데이터·사전·optimizer·스케줄 변경은 새 실험으로 기록한다. [save/load 구현](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/utils/save_load.py)

### 6.7 추론 모델 export

실행 위치: 동일. 입력: val로 선택한 **실제 checkpoint prefix**, 완성 YAML. 출력: 새 `training/exports/korie_rec_v1/`. 성공 기준: export 종료 성공, graph/parameter/config 생성, 사전 및 전처리 일치, 6.5절 재평가 통과. 선택이 완료되지 않았으므로 `SELECTED`를 자동으로 best_accuracy에 지정하지 않는다.

```bash
read -r -p 'Validation으로 선택한 checkpoint prefix: ' SELECTED
test -f "${SELECTED}.pdparams" || exit 1
printf '%s\n' "$SELECTED" > ../records/selected_checkpoint.txt
python tools/export_model.py -c ../configs/korie_rec.yml -o "Global.pretrained_model=$SELECTED" Global.save_inference_dir=../exports/korie_rec_v1
find ../exports/korie_rec_v1 -maxdepth 2 -type f
sha256sum ../configs/korie_rec.yml > ../records/final_config.sha256
```

export는 일반적으로 `inference.json`, `inference.pdiparams`, `inference.yml`을 만들지만 실제 출력 파일/형식은 선택한 Paddle export 실행에서 확인한다. rename만으로 추론 모델을 만들지 않는다. [공식 export 진입점](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/tools/export_model.py)

### 6.8 기존 검출기와 전체 영수증 테스트 — 추가 확인/구현 필요

실행 위치 계획: 프로젝트 루트. 입력: 기존 검출 모델의 고정 가중치/옵션, 전체 영수증 이미지 및 대응 GT, baseline/튜닝 rec export. 출력 계획: `training/evaluation/full_receipt/<run_id>/`에 bbox, crop, 인식 문자열, 읽기 순서, 필드 결과, 실패 유형.

현재 KORIE crop만으로 이 테스트를 실행할 수 없다. 유지할 검출기와 원본 영수증·GT 연결을 확보한 뒤 **같은 검출 출력 crop에 두 인식기를 적용**하고, 고정한 end-to-end 파이프라인의 결과도 별도 평가한다. 검출 threshold, 회전/기하 보정, 정렬, 필드 추출을 함께 바꾸지 않는다. 개인 영수증 라벨의 자동 OCR 좌표와 수정 전사는 동일 단위 대응이 보장되지 않으므로 전체 OCR 정답으로 곧바로 취급하지 않는다.

성공 기준: detector 모델 해시 동일, 입력 이미지 동일, 누락/과검출과 recognition 오류 구분, raw crop 평가와 end-to-end 지표 별도 보고. 기존 검출기의 이름/경로가 확인되기 전에는 임의 `PaddleOCR()` 기본 모델을 사용하는 명령을 제시하지 않는다. 서버 배포는 export 재평가 후 서버 OS/런타임에서도 smoke·정확도 동등성·메모리/지연을 실제 측정한 다음 별도 진행한다.

## 7. 평가 규칙

공식 KORIE 평가의 세부 구현은 아직 확인 필요다. 첫 비교는 로컬 baseline과 아래 규칙을 동일하게 유지한다.

| 지표 | 정의 |
|---|---|
| CER % | `100 × Σ Levenshtein(NFC(GT), NFC(pred)) / Σ len(NFC(GT))`; crop별 비율 평균이 아님 |
| 문자열 Exact Match % | NFC 후 문자열 전체 일치 수 / 전체 평가 crop 수 × 100 |
| 숫자·기호 문자열 EM % | 기존 `metrics.py`의 numeric_symbol_only 집합에서 EM; 숫자 ≥1, Unicode 이름 LATIN/HANGUL인 letter 없음 |
| 금액 EM % | 필드명이 금액임을 확인한 별도 집합에서 문자열 EM; 숫자·기호 전체와 구분 |
| WER % | 기존 결과와 대조할 때 보조 기록; `str.split()` 토큰의 micro 편집거리 |

숫자 집합은 날짜·전화·바코드도 포함한다. 구현은 모든 언어의 letter를 제외하는 조건이 아니므로 표현을 임의로 넓혀 설명하지 않는다. 금액 필드는 파일명에 `Total`, `Subtotal`, `TotalTax`, `Item_Unit_Price`, `Item_Total_Price` 등이 실제 존재한다. 의미가 확인된 whitelist를 **val에서 고정**하고 필드별 n과 EM을 먼저 보고한다. discount/Tip 등의 포함 여부도 기록한다. 쉼표 삭제/통화 변환을 하는 “금액 수치 일치”는 이 문자열 EM과 다른 지표다.

비교 시 같은 이미지 바이트/목록/GT/사전·전처리 정책/정규화/장치 조건을 사용한다. 사전 OOV 때문에 baseline에는 어려운 샘플을 빼고 튜닝 모델에만 쉬운 샘플을 남기지 않는다. 모델별 실패는 누락과 분리 기록한다. 정답 crop 성능은 검출 실패가 포함되는 전체 OCR 성능과 다른 표에 보고한다.

개선/악화 표는 `image_path + image_sha256`로 매칭하고 원문 GT·두 예측·편집거리 변화·field·길이/OOV 플래그를 포함한다. `worst_50.csv`만 보면 전체 변화가 왜곡될 수 있으므로 전체 paired 결과를 집계한다. 새로운 정확도, GPU 시간, epoch 소요 시간은 **실측 전 공란**으로 둔다.

## 8. 오류 대응

| 증상 | 확인 및 대응 |
|---|---|
| Python/venv 실행 불가 | `.venv/pyvenv.cfg`의 base Python과 `py -0p`를 확인. 기존 폴더 존재를 설치 성공으로 취급하지 않고 독립 환경 준비 |
| GPU 미인식 | `nvidia-smi`, CUDA 빌드 여부, GPU count, 사용 Python, CPU/GPU 배포명 점검. WSL이면 Linux 내부에서 GPU 전달 확인 |
| `no kernel image`, unsupported architecture | RTX 50용 cu129 후보인지 확인. Windows recognition 학습 제한을 고려해 Linux/WSL2 검증. 구형 cu118/cu126을 UMD 숫자만 보고 선택하지 않음 |
| CUDA/cuDNN DLL 또는 package 충돌 | `pip check`, `pip freeze`, 실제 import 경로, PATH/LD_LIBRARY_PATH 확인. baseline 환경에 덮어쓰지 않고 깨끗한 학습 venv에서 한 Paddle 패키지만 설치 |
| 기본 역전파 성공, OCR backward 실패 | convolution/CTC/NRTR stack trace와 정확한 wheel/driver 기록. 전체 OCR smoke 실패 상태로 본 학습 진행 금지 |
| 메모리 부족 | sampler first_bs와 loader batch 모두 축소, 다른 GPU 프로세스 확인, num_workers=0 유지. 폭/증강 변경은 기록 후 별도 비교. 정확도 기준 통과 전 AMP 추가하지 않음 |
| 사전/출력 head mismatch | 정확한 한국어 v5 사전·순서·공백·checkpoint 종류 점검. 경고를 무시한 partial load를 성공으로 판정하지 않음. 사전을 맞추려 head를 임의 재초기화하지 않음 |
| 길이 초과/OOV | raw/NFC 길이와 effective dictionary 비교, 샘플 ID·빈도 기록. 잘라내기/문자 제거 금지. 내장 evaluator의 제외와 전체 raw 평가 분모를 구분 |
| data loader 오류·반복 정지 | root+상대경로, TAB/UTF-8, 이미지 decode, 빈 label, worker=0에서 최초 오류 확인. loader가 잘못된 샘플을 다른 샘플로 대체하는지 검증 |
| loss 감소, val 악화 | train 과적합/학습률/증강/전처리 차이 점검. 이전 val 최적 checkpoint로 선택, 더 짧은 학습·낮은 LR을 train/val에서만 비교. test로 조건 조정 금지 |
| export 후 성능 변화 | 같은 sample ID의 예측 비교, CTC decoder/사전/폭/precision/graph 버전 확인. 문제 해결 전 서버 배포 보류 |

## 9. 실험 기록 양식

실험마다 새 run ID를 사용한다. 기록 파일과 모델·데이터를 기존 baseline 위치에 덮어쓰지 않는다. 현 프로젝트는 유효 HEAD가 확인되지 않아 최초에는 해당 사실과 관련 코드 SHA-256을 기록하고, 추후 commit이 생기면 그 SHA와 dirty 상태를 기록한다. upstream v3.7.0 tag만 적지 말고 실제 clone commit을 저장한다.

| 항목 | 기록값 |
|---|---|
| run ID / 시작·종료 시각 / 실행자 | |
| 학습 OS/WSL 배포판, CPU/RAM/GPU/드라이버 | |
| Python 경로·patch 버전, Paddle wheel URL·버전, CUDA build | |
| PaddleOCR commit / 프로젝트 commit 또는 미생성 / 코드 해시 | |
| pip freeze / pip check 결과 | |
| KORIE 출처·버전·ZIP hash(미확보 시 확인 필요) | |
| 공식 split별 입력 수 / 사용 수 / 보류 수 / 사유 | |
| 원본 manifest·이미지·라벨 해시 / 변환기 버전 | |
| raw→NFC 정책, 변경 수 / OOV / 길이 분포 | |
| 사전 경로·SHA-256·순서 대조 결과 | |
| pretrained URL·SHA-256 / 완성 YAML·SHA-256 | |
| seed / LR / batch(first_bs 포함) / max_text_length / 폭 | |
| 증강 / optimizer / precision / workers | |
| epoch별 train loss·전체 val CER/EM / 최적 epoch | |
| selected checkpoint prefix·hash / export 파일 hash | |
| test 동결 설정·표식 / 평가기 버전 / 금액 필드 whitelist | |
| 실제 경과 시간 / peak VRAM / 서버 smoke 결과 | |

새 실험 결과 기록표(기존 실측치는 1.2절에 별도 보존):

| run | split | 모델/epoch | n | CER % | EM % | 숫자·기호 n/EM % | 금액 n/EM % | 실패 수 | 실제 시간 |
|---|---|---|---|---|---|---|---|---|---|
| | val | 동일 환경 baseline | | | | | | | |
| | val | 튜닝 checkpoint | | | | | | | |
| | val | 튜닝 export | | | | | | | |
| | test | 동결 baseline | | | | | | | |
| | test | 동결 튜닝 | | | | | | | |

## 10. 실행 전에 남은 확인·구현

- **환경 확인 필요:** 실제 학습 OS, 작동하는 Python 3.11, WSL2 GPU 사용 가능 여부, 선택 wheel의 실제 OCR backward 호환성.
- **추가 확보 필요:** v3.7.0 학습 checkout/commit, 공식 한국어 pretrained와 해시, 학습 사전 대조.
- **추가 구현 필요:** 공식 split 보존 목록 변환, 실제 decode/OOV/길이·폭 분포 검사와 보고, 전체 val/test raw GT 기반 튜닝·export 비교 평가기.
- **확인 필요:** 학습용 가중치와 기존 HF 추론 baseline 대응, 최종 max_text_length·폭·배치, 공식 KORIE 평가 상세 규칙·데이터 버전.
- **전체 영수증 단계 확인 필요:** 유지할 검출 모델과 고정 옵션, 원본 이미지·annotation 연결, end-to-end 평가 진입점.

원본 데이터/라벨 수정, 임의 재분할, 전체 수동 검수는 위 항목을 해결하기 위한 필수 조건이 아니다.
