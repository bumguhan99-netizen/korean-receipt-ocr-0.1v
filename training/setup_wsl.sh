#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$TASK_ROOT/records"
export UV_UNMANAGED_INSTALL="$TASK_ROOT/bootstrap/bin"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/bootstrap/python"
export UV_CACHE_DIR="$TASK_ROOT/bootstrap/cache"
sh "$TASK_ROOT/bootstrap/uv-install.sh"
UV="$TASK_ROOT/bootstrap/bin/uv"
"$UV" python install 3.11
"$UV" venv --python 3.11 --seed "$TASK_ROOT/.venv"
PY="$TASK_ROOT/.venv/bin/python"
"$PY" -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git "$TASK_ROOT/PaddleOCR"
"$PY" -m pip install -r "$TASK_ROOT/PaddleOCR/requirements.txt"
"$PY" -m pip install paddleocr==3.7.0
"$PY" -m pip check
git -C "$TASK_ROOT/PaddleOCR" rev-parse HEAD > "$TASK_ROOT/records/paddleocr_commit.txt"
"$PY" -m pip freeze > "$TASK_ROOT/records/packages.txt"
mkdir -p "$TASK_ROOT/pretrained"
curl -fL --retry 2 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams' -o "$TASK_ROOT/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
sha256sum "$TASK_ROOT/pretrained/"*.pdparams > "$TASK_ROOT/records/pretrained.sha256"
