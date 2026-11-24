# 한국어 영수증 데이터셋

## 프로젝트 탐색

### 설치 환경과 GitLab 저장 범위

루트 `requirements.txt`는 개인 영수증 유틸리티의 의존성과 환경별 설치 파일 안내를 담습니다. OCR baseline은 `ocr_baseline/requirements.txt`, GLM은 `ocr_baseline/requirements-glm.txt`, WSL GPU 학습은 `training/requirements.txt`를 각각 독립된 가상환경에서 사용합니다. 학습용 GPU wheel은 해당 파일의 주석대로 먼저 설치합니다. 각 환경의 `requirements*.lock.txt`는 기존 설치 버전 기록이며 이번 작업에서 재설치 검증하지 않았습니다.

`.gitignore`는 원본 영수증·데이터·라벨·열람 CSV·사용자 확인 기록(`check.md`), 가상환경·다운로드 캐시·모델·학습 결과·로그·로컬 비밀 설정을 제외합니다. `dataset/scripts/`의 유틸리티와 학습 코드·설정·문서·requirements는 포함합니다. 따라서 GitLab에서 clone한 뒤에는 데이터와 모델을 별도 공급해야 하며, 아래의 로컬 데이터 링크는 해당 파일을 복원해야 열립니다. 실제 인증정보는 `.env.example`에 넣지 마세요.

| 위치 | 용도 |
|---|---|
| [파인튜닝 실행 가이드](docs/KORIE_PP-OCRv5_GPU_파인튜닝_실행.md) | KORIE 학습 환경·준비·실행 절차 |
| [모델 선정 및 데이터 계획](docs/모델선정_및_데이터계획.md) | 기존 실험 메모(내용 보존) |
| [영수증 목록](reports/receipts/영수증_목록.csv), [줄별 내용](reports/receipts/영수증_줄별내용.csv) | 열람용 CSV 결과 |
| [사용자 확인 기록](check.md) | 라벨에 기록된 출처 경로 유지 |
| `scripts/` | 개인 영수증 준비·검수·CSV 생성 코드 |
| `dataset/` | 데이터·라벨·분할·검수 기록 |
| `receipt/`, `datasets--HumynLabs--Korean_Receipts_Dataset/` | 원본 이미지 및 다운로드 캐시 |
| [ocr_baseline/](ocr_baseline/README.md) | OCR 비교 코드·모델·평가 결과 |
| `training/` | 학습 코드·설정·체크포인트·로그·export |
| `venv/` | 기존 로컬 Python 가상환경 |

문서 안의 실행 명령과 코드 경로는 별도 안내가 없으면 프로젝트 루트 기준입니다. 데이터·모델·가상환경과 실행 로그는 기존 참조 경로를 유지합니다. 파일 이동 내역은 [배치 안내](docs/파일배치.md)를 참고하세요.


KORIE의 두 PP-OCRv5 인식 모델 비교 실험은 별도 폴더 [ocr_baseline](ocr_baseline/README.md)에서 진행합니다. 아래 개인 영수증 데이터와 혼합하지 않습니다.

이미지와 라벨 파일명은 `001`부터 `052`까지 같은 번호로 맞췄습니다. 기존 CSV의 1~52번 순서를 유지합니다.

- 이미지: `dataset/images/001.jpg`
- JSON 라벨: `dataset/labels/001.json`
- 확인 목록: `reports/receipts/영수증_목록.csv`의 1번 행

목록 열은 `번호, 이미지, 라벨, 상호, 날짜, 합계, 결제액, 내용, 메모`입니다. 원본 이름과 내부 ID는 `dataset/filename_mapping.csv`에 남아 있습니다. 재생성할 때도 `dataset/name_index.json`에 저장된 번호를 유지합니다.

## 이미지 대조 후 수정한 CSV

`reports/receipts/`의 `reports/receipts/영수증_목록.csv`, `reports/receipts/영수증_줄별내용.csv`는 52장 이미지를 AI가 직접 대조하여 **상호·거래일·상품·수량·금액 중심으로 다시 옮긴 내용**입니다. 광고·반품 안내·주소·카드번호 등을 포함한 영수증 전체 전사는 아닙니다. 판독 방법과 범위는 `dataset/visual_corrections.json`에 기록했습니다. 사람의 최종 검수가 완료된 학습 정답은 아닙니다.

- `합계`: 영수증에 표기된 합계(원). 할인 전 금액일 수 있습니다.
- `결제액`: 실제 결제/승인 금액(원)이 보이는 경우만 입력했습니다.
- 금액 또는 날짜 공란은 0이 아니라 확인 불가, 미기재 또는 복수 전표입니다. 이유는 `메모`를 확인하세요.
- 37번 이미지는 서로 다른 거래 2건으로 날짜와 합계를 하나로 합치지 않았습니다. 34번은 동일 거래의 전표 2장으로 중복 합산하지 않았습니다.
- 판독이 어려운 글자는 `판독 불가`로 표시했습니다. 원본의 약어·상품 표기는 보존합니다.
- 수정 근거는 `dataset/visual_corrections.json`, 사용자 확인값은 `dataset/user_corrections.json`, 원래 OCR CSV는 `dataset/ocr_original_exports/`에 있습니다. JSON 라벨과 학습 JSONL의 `fields` 및 `transcription`에도 수정 내용을 반영했습니다. `ocr` 텍스트·좌표는 자동 초안으로 보존했으며 수정된 전사와 좌표가 대응한다고 간주하면 안 됩니다. 수정 전 라벨은 `dataset/backups/before_core_label_sync/`에 보관했습니다.

사용자 확인값을 변경할 때는 `dataset/user_corrections.json`을 수정하세요. `check.md`는 확인 기록이며 자동 파싱하지 않습니다. 수정 내용을 재생성하는 순서는 다음과 같습니다. 이미지 ID와 원본 해시를 검증하며, 최종 검수 상태인 라벨은 덮어쓰지 않습니다.

```powershell
python scripts/visual_corrections.py
python scripts/sync_visual_labels.py
python scripts/export_readable_csv.py
python scripts/validate_dataset.py
```

원본 `receipt/` 및 Hugging Face 캐시 이미지를 보존하고 `dataset/`에 정규화된 데이터셋을 구성했습니다. OCR 학습용 **자동 초안 라벨**이 포함됩니다. 사람이 검수한 정답 데이터셋은 아닙니다. 모든 이미지가 영수증이므로 비영수증 분류 학습에는 별도 음성 샘플이 필요합니다.

## 파일 구성

- `dataset/images/001.jpg` 등: EXIF 회전 적용, RGB/JPEG, 긴 변 최대 2400px. 내부 ID는 원본 상대 경로의 SHA-256 앞 16자리로 재현됩니다.
- `dataset/labels/*.json`: 이미지별 전체 OCR 텍스트, 줄/단어 및 단어 좌표, 검수 상태.
- `dataset/{train,validation,test}.jsonl`: 이미지 메타데이터와 라벨을 함께 담은 학습용 레코드. 경로는 `dataset/` 기준입니다.
- `dataset/splits/*.txt`: 각 분할에 속한 ID.
- `dataset/filename_mapping.csv`: 원본 파일명 ↔ 새 파일명 대응표.
- `dataset/manifest.json`: 원본 SHA-256, 크기, 출처, 분할, 그룹, 품질 플래그.
- `dataset/quality_report.json`: 실제 수량, 오류, 중복/유사 후보, 분할 통계.
- `dataset/contact_sheet.jpg`: 전체 이미지 미리보기.

## 라벨 규칙과 한계

`document_type=receipt`는 수집 폴더와 이미지 확인에 따른 문서 종류입니다. `ocr`는 Windows 한국어 OCR 출력이며 confidence는 엔진이 제공하지 않아 null입니다. `bbox=[x_min,y_min,x_max,y_max]`는 **정규화된 JPG 이미지 기준 픽셀 좌표**입니다. 원본 좌표가 아닙니다. 줄 순서와 인식 오류는 검수해야 합니다. 회전된 글자와 영수증 일부만 찍힌 이미지가 포함되어 있습니다.

`annotation_status=auto_labeled_needs_review`는 전체 OCR이 아직 자동 초안임을 의미합니다. 실제 전체 대조 검수를 마친 것만 `reviewed`로 변경하세요. `fields`의 상호·날짜·합계·결제액·통화와 `transcription.text`는 이미지 대조 내용 및 사용자 확인값을 반영했습니다. `fields_status`와 `transcription.user_confirmation`으로 검수 범위를 구분합니다. null은 0이 아니며 `field_status`의 `not_present`(미기재), `cropped_out`(잘림), `multiple_receipts`(복수 거래) 및 전사 메모를 확인하세요. `items_status=not_structured`인 품목 배열은 아직 미구조화 상태입니다. 37번 거래별 품목은 `documents`에 있습니다. 30번의 잘린 금액은 이미지 판독값이 아닌 사용자 확인값입니다.

80/10/10을 목표로 seed 42로 분할합니다. 원본 바이트 중복, dHash 거리 ≤6과 유사 종횡비, 날짜 파일명의 연속 촬영 변형을 같은 그룹에 넣습니다. 그룹 단위를 유지하므로 실제 비율은 달라집니다. 유사도는 후보 탐지용이며 동일 거래 여부를 보장하지 않습니다. 촬영 각도가 크게 다른 같은 영수증은 추가 수동 검수가 필요합니다. 원본은 삭제하거나 중복 제거하지 않습니다.

## 실행과 검수

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/validate_dataset.py
python scripts/review_labels.py
```

검수 화면: <http://127.0.0.1:8765>. 이미지 옆 JSON을 수정해 저장합니다. 직전 라벨은 `.json.bak`으로 보관합니다. 저장 후 아래 명령으로 JSONL을 다시 생성하세요.

```powershell
python scripts/prepare_dataset.py --skip-ocr
python scripts/validate_dataset.py
```

**일반 재실행은 OCR을 덮어쓰므로 사람의 수정 이후에는 `--skip-ocr`를 사용하세요.** 이미지 처리에는 Python/Pillow, OCR에는 Windows와 설치된 한국어 OCR 언어팩이 필요합니다. 외부 OCR 서비스에 이미지를 전송하지 않습니다.

학습 데이터 읽기:

```python
import json
from pathlib import Path
from PIL import Image
root = Path('dataset')
for line in (root / 'train.jsonl').read_text(encoding='utf-8').splitlines():
    sample = json.loads(line)
    image = Image.open(root / sample['image'])
    text = sample['annotation']['ocr']['text']
```

## 출처 및 공개 범위

`HumynLabs/Korean_Receipts_Dataset`, snapshot `656cef561f4b8ff22e46d6785bc2807e78af99a1`: 로컬 원본 README에 CC BY 4.0이 명시되어 있습니다. 저작자 표기와 라이선스 고지를 유지하고, 이미지 정규화 및 OCR 라벨 추가 사실을 밝혀야 합니다. 원본 고지는 캐시 snapshot의 README.md를 참조하세요. 원본 설명의 규모와 무관하게 여기서는 실제 로컬 파일만 처리합니다.

`receipt/` 파일의 사용권은 제공되지 않아 `unspecified`로 기록했습니다. 전체 결과에 단일 라이선스를 부여하지 않습니다. 개인정보 마스킹을 수행하지 않았으며 `privacy_review_status=pending`입니다. 전화번호·카드 정보 등 이미지와 OCR 텍스트를 함께 확인해야 합니다. 현재 결과는 로컬 작업용이며 외부 공개를 수행하지 않았습니다.
