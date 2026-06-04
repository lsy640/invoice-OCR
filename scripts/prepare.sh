#!/bin/bash
# 计算节点准备步骤（计算节点有外网）：安装依赖 + 规范化数据 + 下载图片。
# 设计为被 source：调用前需已激活 env、cd 到 OCR 项目根、并设好 HF_HOME。
# 幂等：依赖已装则跳过；图片已存在则跳过；可断点续跑。
# 模型权重不在此显式下载——vllm serve 启动时自动从 HF 拉取并缓存到 HF_HOME。

echo "==> [prepare] 检查/安装数据层依赖"
# 仅检查数据准备所需（推理框架 vllm/transformers 由各自 env 提供）
if ! python -c "import yaml,PIL,tqdm,pandas,openpyxl,requests" 2>/dev/null; then
    pip install pyyaml pillow tqdm pandas openpyxl requests
else
    echo "    依赖已就绪，跳过安装"
fi

echo "==> [prepare] 规范化数据 -> data/records.json"
python src/prepare_data.py ${LIMIT:+--limit "$LIMIT"}

echo "==> [prepare] 下载图片到 data/images（幂等续跑）"
python src/download_images.py --workers "${IMG_WORKERS:-16}"
