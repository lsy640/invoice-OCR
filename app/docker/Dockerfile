# GLM-OCR 票据/停车识别 — 后端镜像（Windows[+NVIDIA] / Linux）
# 说明：torch 的 CUDA wheel 自带 CUDA 运行库；配合 NVIDIA Container Toolkit + `--gpus all`
# 即用 GPU，否则自动回退 CPU（代码按 torch.cuda.is_available() 选择）。
# macOS 的 MLX 加速无法在 Docker 内使用（容器是 Linux VM），请改用 app/scripts/run_macos_mlx.sh。
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    INFERENCE_BACKEND=auto

WORKDIR /workspace

# OpenCV/Pillow 等所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# 构建上下文为仓库根目录 OCR/（见 docker-compose.yml）
COPY app/requirements-backend.txt /workspace/app/requirements-backend.txt
RUN pip install --upgrade pip && pip install -r /workspace/app/requirements-backend.txt

COPY src/ /workspace/src/
COPY app/ /workspace/app/
COPY config.yaml /workspace/config.yaml

EXPOSE 8000
# 健康检查：模型惰性加载，/api/health 立即可用
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["python", "app/backend/server.py"]
