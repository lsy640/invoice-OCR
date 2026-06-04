#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=ocr_park
#SBATCH --output=OCR/results/output_%x_%j.out
#SBATCH --error=OCR/results/error_%x_%j.err
#SBATCH --signal=B:USR1@300
#
# 停车数据：GLM-OCR 逐图抽取(进出场水印 + 支付凭证) -> 聚合(时长/单价) -> 导出 Excel。
# transformers 后端，conda env `env`。图片走公有 OBS 直链（无需 token）。
#
# 提交（在 /home/msai/lius0131 下）：  sbatch OCR/scripts/run_parking_glm.sh
# 小样本调试：  sbatch --export=ALL,LIMIT=5 OCR/scripts/run_parking_glm.sh

set -o pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/msai/lius0131}"
PROJECT_ROOT="${OCR_PROJECT_ROOT:-/home/msai/lius0131/OCR}"
cd "$PROJECT_ROOT" || { echo "找不到项目目录 $PROJECT_ROOT"; exit 1; }

CONDA_ENV="env"
HF_HOME_CFG="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['paths']['hf_home'])")"

nvidia-smi || true
if command -v module &>/dev/null; then module load anaconda || true; fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME_CFG}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results "${HF_HOME}"

SCRIPT_PATH="$(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -oP 'Command=\K\S+')"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/scripts/run_parking_glm.sh}"

resubmit() {
    echo "$(date): 收到超时信号，终止并重新提交续跑..."
    kill "${EVAL_PID}" 2>/dev/null || true
    wait "${EVAL_PID}" 2>/dev/null || true
    ssh CCDS-TC2 "cd ${SUBMIT_DIR} && sbatch ${SCRIPT_PATH}" || echo "!! 重提交失败，请手动 sbatch ${SCRIPT_PATH}"
    exit 0
}
trap 'resubmit' USR1

# ── 准备：依赖 + 规范化 + 下载图片（计算节点有外网，OBS 直链）──
echo "==> 检查数据层依赖"
python -c "import yaml,PIL,tqdm,pandas,openpyxl,requests" 2>/dev/null || pip install pyyaml pillow tqdm pandas openpyxl requests
echo "==> 规范化停车数据"
python src/parking_prepare.py ${LIMIT:+--limit "$LIMIT"}
echo "==> 下载停车图片(OBS 直链, 幂等续跑)"
python src/parking_download.py --workers 16

# ── 抽取 + 聚合 ───────────────────────────────────────────────────
echo "==> GLM-OCR 逐图抽取"
python src/parking_extract_hf.py ${LIMIT:+--limit "$LIMIT"} ${WORKFLOWS:+--workflows "$WORKFLOWS"} &
EVAL_PID=$!
wait "${EVAL_PID}"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "==> 聚合 + 导出 Excel"
    python src/parking_aggregate.py
    echo "$(date): 停车任务完成"
else
    echo "$(date): 抽取异常退出 (exit code: ${EXIT_CODE})"
fi
