#!/usr/bin/env bash
# Single use RTX 5060 Ti experiment, official architecture/dictionary preserved.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$TASK_ROOT")"
PY="$TASK_ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [ -d /usr/lib/wsl/lib ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "$TASK_ROOT/records" "$TASK_ROOT/configs" "$TASK_ROOT/evaluation"
cd "$TASK_ROOT"
"$PY" -m unittest -v test_korie 2>&1 | tee records/tests.log
"$PY" gpu_check.py 2>&1 | tee records/gpu_check.log
if [ ! -d data ]; then
  "$PY" korie.py prepare --dictionary PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt --output data 2>&1 | tee records/data_prepare.log
fi
"$PY" korie.py verify --data data 2>&1 | tee records/data_verify.log
"$PY" verify_encoder.py 2>&1 | tee records/encoder_check.log
"$PY" korie.py config --repo PaddleOCR --data data --weights pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams --output configs/official.yml --output-dir outputs/official
"$PY" korie.py config --repo PaddleOCR --data data --weights pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams --output configs/smoke.yml --output-dir outputs/smoke --smoke
cd "$TASK_ROOT/PaddleOCR"
"$PY" tools/train.py -c ../configs/smoke.yml 2>&1 | tee ../records/smoke.log
test -f ../outputs/smoke/latest.pdparams
# Fail rather than proceed after an incomplete/incompatible pretrained load.
if grep -Ei 'shape.*not matched|not in loaded params|nan|no kernel image' ../records/smoke.log; then
  echo 'Smoke requires investigation; full training not started.'
  exit 1
fi
"$PY" tools/export_model.py -c ../configs/official.yml -o Global.save_inference_dir=../exports/initial 2>&1 | tee ../records/export_initial.log
"$PY" ../korie.py evaluate --manifest ../data/manifest.jsonl --baseline "$PROJECT/ocr_baseline/models/korean_PP-OCRv5_mobile_rec" --candidate ../exports/initial --output ../evaluation/initial_val 2>&1 | tee ../records/initial_val.log
"$PY" ../tune_resources.py 2>&1 | tee ../records/resource_tuning.log
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.gr,clocks.mem,temperature.gpu --format=csv -l 2 > ../records/training_gpu.csv &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PY" tools/train.py -c ../configs/selected_resources.yml 2>&1 | tee ../records/train.log
kill "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
test -f ../outputs/official/best_accuracy.pdparams
"$PY" tools/eval.py -c ../configs/selected_resources.yml -o Global.checkpoints=../outputs/official/best_accuracy 2>&1 | tee ../records/native_selected_val.log
"$PY" tools/export_model.py -c ../configs/selected_resources.yml -o Global.pretrained_model=../outputs/official/best_accuracy Global.save_inference_dir=../exports/selected 2>&1 | tee ../records/export_selected.log
"$PY" ../korie.py evaluate --manifest ../data/manifest.jsonl --baseline ../exports/initial --candidate ../exports/selected --output ../evaluation/selected_val 2>&1 | tee ../records/selected_val.log
"$PY" ../korie.py freeze --run ../evaluation/selected_val --output ../evaluation/frozen_test.json
"$PY" ../korie.py evaluate --manifest ../data/manifest.jsonl --baseline ../exports/initial --candidate ../exports/selected --split test --frozen ../evaluation/frozen_test.json --output ../evaluation/final_test 2>&1 | tee ../records/final_test.log
sha256sum ../exports/initial/* ../exports/selected/* > ../records/export_hashes.txt
"$PY" ../korie.py report --output ../RESULTS.md
echo 'Completed RTX 5060 Ti fine-tuning, validation and frozen paired test.'
