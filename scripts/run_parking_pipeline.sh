#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=30G
#SBATCH --job-name=ocr_pipe
#SBATCH --output=OCR/results/output_%x_%j.out
#SBATCH --error=OCR/results/error_%x_%j.err
#
# 端到端 pipeline：输入一组图片 -> 直接输出完整识别结果(JSON)。
# 用 IMAGES(逗号分隔URL/路径) 或 IMAGES_FILE 传入；可选 COST / VIN 标签。
#   sbatch --export=ALL,IMAGES_FILE=imgs.txt,COST=80 OCR/scripts/run_parking_pipeline.sh
#   sbatch "--export=ALL,IMAGES=url1;url2;url3,COST=80" OCR/scripts/run_parking_pipeline.sh

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
mkdir -p results

# IMAGES 用分号分隔以避开 --export 的逗号；这里转回逗号给 --images
ARGS=()
if [[ -n "${IMAGES_FILE:-}" ]]; then ARGS+=(--images-file "${IMAGES_FILE}"); fi
if [[ -n "${IMAGES:-}" ]]; then ARGS+=(--images "${IMAGES//;/,}"); fi
[[ -n "${COST:-}" ]] && ARGS+=(--cost "${COST}")
[[ -n "${VIN:-}" ]] && ARGS+=(--vin "${VIN}")
ARGS+=(--out "results/pipeline_result_${SLURM_JOB_ID:-local}.json")

python src/parking_pipeline.py "${ARGS[@]}"
echo "$(date): pipeline 完成"
