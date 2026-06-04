#!/usr/bin/env bash
# 本地直跑后端（Linux / macOS-CPU / WSL）。Windows 见 README_app.md 的 PowerShell 版。
# 用法： bash app/scripts/run_local.sh [transformers|mlx|auto]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> OCR/

PY="${PYTHON:-python3}"
VENV=".venv-app"
if [[ ! -d "$VENV" ]]; then
  echo "==> 创建虚拟环境 $VENV"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r app/requirements-backend.txt
fi

export INFERENCE_BACKEND="${1:-auto}"
echo "==> 启动后端 (backend=$INFERENCE_BACKEND)  http://localhost:8000"
exec "$VENV/bin/python" app/backend/server.py
