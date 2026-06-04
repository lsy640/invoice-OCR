#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --job-name=ocr_park_rf
#SBATCH --output=OCR/results/output_%x_%j.out
#SBATCH --error=OCR/results/error_%x_%j.err
#
# 二次 OCR 补救：对未解析到停车区间的发票图裁剪备注区域+放大重 OCR，回填 pred 后重新聚合。
# 需先有 results/parking_glm_pred.jsonl（即先跑过 run_parking_glm.sh）。
#   sbatch OCR/scripts/run_parking_refine.sh

set -o pipefail
PROJECT_ROOT="${OCR_PROJECT_ROOT:-/home/msai/lius0131/OCR}"
cd "$PROJECT_ROOT" || { echo "找不到项目目录 $PROJECT_ROOT"; exit 1; }
HF_HOME_CFG="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['paths']['hf_home'])")"

nvidia-smi || true
if command -v module &>/dev/null; then module load anaconda || true; fi
eval "$(conda shell.bash hook)"
conda activate env
export HF_HOME="${HF_HOME_CFG}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "==> 发票备注区域二次 OCR"
python src/parking_refine_invoice.py
echo "==> 重新聚合 + 导出 Excel"
python src/parking_aggregate.py
echo "$(date): refine 完成"
