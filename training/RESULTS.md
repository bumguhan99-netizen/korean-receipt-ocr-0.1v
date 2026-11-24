# KORIE GPU 파인튜닝 실행 결과

공식 한국어 v3.7.0 구조·사전, RTX 5060 Ti용 설정. 원본 split 유지.

학습 설정: first_bs 32(멀티스케일 32/21/16), train workers 4, eval workers 2, 최대 30 epochs, LR 1e-5, warmup 1. 선택 규칙: 내장 val acc 최고 checkpoint.

처리량 실측 후 최종 first_bs=32, train workers=12. probe 처리량 82.73 samples/s, peak reserved 4772 MiB. 위 32/4 시작값보다 이 최종 선택값이 우선한다.

| split | 원본 | 학습/내장평가 가능 | 보류 |
|---|---:|---:|---:|
| train | 5356 | 5087 | 269 |
| val | 1785 | 1693 | 92 |
| test | 1786 | 1711 | 75 |

보류는 OOV/길이 등의 사유로 issues.csv에 기록. 외부 val/test는 보류 표본도 포함한다.

| 평가 | 모델 | n | CER % ↓ | EM % ↑ | 숫자·기호 EM % | 금액 EM % |
|---|---|---:|---:|---:|---:|---:|
| initial_val | baseline | 1785 | 30.8679 | 54.5658 | 59.6587 | 50.8197 |
| initial_val | candidate | 1785 | 30.8679 | 54.5658 | 59.6587 | 50.8197 |
| selected_val | baseline | 1785 | 30.8679 | 54.5658 | 59.6587 | 50.8197 |
| selected_val | candidate | 1785 | 23.9574 | 71.7087 | 75.6314 | 76.3206 |
| final_test | baseline | 1786 | 28.3714 | 53.9754 | 58.4647 | 52.1352 |
| final_test | candidate | 1786 | 21.1646 | 71.5566 | 74.7772 | 77.2242 |

initial_val: 기존 HF 모델 vs 공식 학습 가중치의 초기 export. selected_val/final_test: 초기 export vs 선택된 튜닝 export.

금액은 Total/Subtotal/TotalTax/Item_Unit_Price/Item_Total_Price 문자열 EM이며, 숫자·기호 지표에는 날짜·전화번호 등도 포함된다.

전체 영수증 검출+인식 평가는 미실행: 고정 검출기와 대응 GT 확인 필요.

증거: `records/`, `data/validation_report.json`, `evaluation/*/run_info.json`, `evaluation/*/summary.json`, `evaluation/*/*_predictions.csv`.
