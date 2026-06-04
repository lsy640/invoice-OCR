#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=ocr_glm_hf
#SBATCH --output=OCR/results/output_%x_%j.out
#SBATCH --error=OCR/results/error_%x_%j.err
#SBATCH --signal=B:USR1@300
#
# GLM-OCR 全量抽取（transformers 后端，conda env `env`：transformers 5.x + torch）。
# vLLM 0.19.0 锁 transformers<5 无法加载 glm_ocr，故走 transformers 离线推理。
# 超时前 300s 收 USR1 -> 杀进程并 ssh 登录节点重新提交自己，断点续跑（按 workflow_no 跳过）。
#
# 提交（在 /home/msai/lius0131 下）：  sbatch OCR/scripts/run_glm_hf.sh
# 小样本调试：  sbatch --export=ALL,LIMIT=5 OCR/scripts/run_glm_hf.sh

# 注：不用 set -u —— conda 的 activate.d 脚本会引用未设置变量
set -o pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/msai/lius0131}"
PROJECT_ROOT="${OCR_PROJECT_ROOT:-/home/msai/lius0131/OCR}"   # Slurm 在 spool 目录执行，必须用绝对路径
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
cd "$PROJECT_ROOT" || { echo "找不到项目目录 $PROJECT_ROOT"; exit 1; }

MODEL_NAME="glm-ocr"
CONDA_ENV="env"                                   # 注意：用 env（transformers 5.x），非 env_vllm
HF_HOME_CFG="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['paths']['hf_home'])")"

# ── 环境 ──────────────────────────────────────────────────────────
nvidia-smi || true
if command -v module &>/dev/null; then module load anaconda || true; fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME_CFG}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results "${HF_HOME}"

SCRIPT_PATH="$(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -oP 'Command=\K\S+')"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/run_glm_hf.sh}"

# 超时前收到 USR1：杀子进程并 ssh 登录节点重新提交（extract_hf 按 workflow_no 断点续跑）
resubmit() {
    echo "$(date): 收到超时信号，终止并重新提交续跑..."
    kill "${EVAL_PID}" 2>/dev/null || true
    wait "${EVAL_PID}" 2>/dev/null || true
    ssh CCDS-TC2 "cd ${SUBMIT_DIR} && sbatch ${SCRIPT_PATH}" || echo "!! 重提交失败，请手动 sbatch ${SCRIPT_PATH}"
    exit 0
}
trap 'resubmit' USR1

# ── 计算节点准备：依赖/数据/图片（模型由 transformers 自动从 HF 拉取到 HF_HOME）──
source "${SCRIPT_DIR}/prepare.sh"

# ── 抽取 + 评测 ───────────────────────────────────────────────────
echo "==> 运行 GLM-OCR 抽取（transformers）"
python src/extract_hf.py --model "${MODEL_NAME}" ${LIMIT:+--limit "$LIMIT"} &
EVAL_PID=$!
wait "${EVAL_PID}"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "$(date): GLM-OCR 抽取已正常完成"
    cat results/metrics_${MODEL_NAME}.json 2>/dev/null || true
else
    echo "$(date): 异常退出 (exit code: ${EXIT_CODE})"
fi
