# training 확인 결과와 검증 코드 v2

2026-11-17 현재 로컬 파일 기준. 기존 파일은 수정하지 않고 `verify_plan_v2.ps1`을 추가했다.

## 이미 있는 기능

- `training/korie.py prepare`: 공식 crop 분할 유지, 이미지/라벨 읽기, 사전 순서 대조, NFC, OOV/길이 검사, 제외 사유 기록 및 manifest 생성.
- `verify`: 원본 이미지/라벨 해시, 분할별 수량, 생성 목록 무결성 검사.
- `evaluate`: 두 모델의 전처리/사전 동일성 검사, 동일 crop 평가, CER·EM·숫자/기호 EM·필드 기반 금액 EM, 개선/악화 CSV 저장. OOV/길이 보류 표본도 외부 평가에 포함.
- `freeze`: 완료된 validation의 모델/manifest 해시와 평가 설정 고정. test 실행은 고정 설정이 필요하며 재실행 방지 파일을 사용한다.
- `training/gpu_check.py`: Paddle CUDA 인식, 연산, 역전파 및 파라미터 업데이트 검사.
- `training/test_korie.py`: 합성 데이터 기반 검사와 모의 인식 평가 테스트. 현재 환경에서 재실행하지는 않았다.

이 기능들은 다시 구현할 필요가 없다. 전체 영수증 검출·박스 대응 평가는 별도로 필요하다.

## 현재 PC에 있는 것과 없는 것

있음: `exports/selected/inference.json`, `inference.pdiparams`, `inference.yml`, 설정/학습/평가 코드, 패키지 버전 기록 파일, `outputs/official/best_accuracy.states`.

없음: `exports/initial`, `data`, `evaluation`, `records`, `.venv`, `PaddleOCR` 저장소 및 `best_accuracy.pdparams/.pdopt`. states 파일만으로 학습 재개나 baseline 복구를 할 수 없다. selected는 추론용 모델이며 이를 baseline으로도 사용하면 안 된다.

`RESULTS.md`에는 final test CER 28.3714%→21.1646%, EM 53.9754%→71.5566%가 기록돼 있다. 이는 과거 crop 평가 기록이며 현재 PC에서 재측정한 결과가 아니다. 원본 run_info/summary가 없어 현재 로컬 자료만으로 재검증할 수 없다.

`requirements.txt`는 과거 WSL/Linux Python 3.11, Paddle GPU 3.3.0/cu129, PaddleOCR 3.7.0 및 소스 commit을 기록한다. 현재 Windows/RTX 5070 Ti 환경의 설치·동작 확인을 의미하지 않는다. setup/run 스크립트는 설치·다운로드·학습까지 실행하므로 사전 검증 목적으로 실행하지 않는다.

## v2 실행

```powershell
Set-Location D:\ssafy\ocr\mogumogu
powershell -NoProfile -ExecutionPolicy Bypass -File .\full_receipt_20261117\verify_plan_v2.ps1
```

현재 PC의 PS1 실행 정책 때문에 해당 프로세스에만 Bypass를 지정한다. 영구 정책은 변경하지 않는다. v2는 기존 검사에 training 실행 자료의 존재/해시 검사를 추가한다. 초기 export가 실제 존재할 때만 baseline 기본값으로 선택하며, 누락 시 임의 모델로 대체하지 않는다. 출력은 새 `verification/<시각_UUID>/report.json`에 저장한다.

다음 단계는 과거 환경의 초기 export·manifest·crop 원본·평가 기록을 복구하거나 공식 원본 자료를 확보하는 것이다. 이후 crop 평가는 기존 `korie.py`를 재사용하고, 전체 영수증은 확인된 annotation에 맞춰 검출/좌표 대응을 추가한다. 이 문서에는 누락된 파일을 현재 실행 가능한 입력처럼 제시하지 않았다.
