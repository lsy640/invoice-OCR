#!/usr/bin/env bash
# macOS (Apple M 系列) 原生跑后端 + MLX 加速（不走 Docker——容器无法访问 Metal/MLX）。
# 用法： bash app/scripts/run_macos_mlx.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> OCR/

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "本脚本仅用于 macOS。其它平台请用 run_local.sh 或 Docker。" >&2; exit 1
fi

PY="${PYTHON:-python3}"
VENV=".venv-mlx"
if [[ ! -d "$VENV" ]]; then
  echo "==> 创建虚拟环境 $VENV 并安装依赖（含 mlx-vlm）"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  # 后端依赖（torch 在 mac 上仅用于少量工具，可装 CPU 版）+ MLX 推理栈
  "$VENV/bin/pip" install -r app/requirements-backend.txt
  "$VENV/bin/pip" install mlx mlx-vlm
fi

# 首次需把 GLM-OCR 转成 MLX 权重（若 mlx-community 无现成版）：
#   "$VENV/bin/python" -m mlx_vlm.convert --hf-path zai-org/GLM-OCR -q --mlx-path ./models/GLM-OCR-mlx
#   并在 config.yaml 设 inference.mlx_repo: "./models/GLM-OCR-mlx"
# 若 mlx-vlm 不支持 GlmOcr 架构，后端会自动回退 transformers-CPU。

export INFERENCE_BACKEND=mlx
echo "==> 启动后端 (MLX)  http://localhost:8000"
exec "$VENV/bin/python" app/backend/server.py
