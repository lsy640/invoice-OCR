#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=30G
#SBATCH --job-name=ocr_e2e
#SBATCH --output=OCR/results/output_%x_%j.out
#SBATCH --error=OCR/results/error_%x_%j.err
set -o pipefail
PROJECT_ROOT="/home/msai/lius0131/OCR"; cd "$PROJECT_ROOT"
HF_HOME_CFG="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['paths']['hf_home'])")"
module load anaconda || true; eval "$(conda shell.bash hook)"; conda activate env
export HF_HOME="${HF_HOME_CFG}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
nvidia-smi || true

python app/backend/server.py > results/_e2e_server.log 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null" EXIT
echo "==> 等待服务起来"
for i in $(seq 1 60); do curl -sf localhost:8000/api/health >/dev/null && break; sleep 2; done
curl -s localhost:8000/api/health; echo

INV_URL="https://obs-prod-biz.obs.cn-east-3.myhuaweicloud.com/changan/9a07ffe3e81843da85b45af9484a206f.png"
PARK_URLS="https://ptmapi.cacxtravel.com/thirdparty/huaweiyun/obs/download/76e554e8a5924831afd29808408631b7.png,https://ptmapi.cacxtravel.com/thirdparty/huaweiyun/obs/download/0cca96d2c23e4f788c05daaf3cedc71b.jpg,https://ptmapi.cacxtravel.com/thirdparty/huaweiyun/obs/download/210573c1a92f423389947fa0635f8ff9.png,https://ptmapi.cacxtravel.com/thirdparty/huaweiyun/obs/download/3b3d3c5e6aad44f89de9dd3c37755149.png"

echo "==> [1] 发票 mode=urls（仅金额，标签54.33）"
curl -s -F task=invoice -F mode=urls -F "urls=${INV_URL}" localhost:8000/api/recognize | python -m json.tool

echo "==> [2] 停车 mode=urls（一组4图，含发票，cost=500/vin=LS6BME2P6SA753459）"
curl -s -F task=parking -F mode=urls -F "urls=${PARK_URLS}" localhost:8000/api/recognize | python -m json.tool

echo "==> [3] 发票 mode=excel（带标签，制造一条不匹配以验证标红）"
python - <<'PY'
import pandas as pd
pd.DataFrame({"pic_url":["https://obs-prod-biz.obs.cn-east-3.myhuaweicloud.com/changan/9a07ffe3e81843da85b45af9484a206f.png"],
             "supply_money":[99.99]}).to_excel("results/_e2e_inv.xlsx", index=False)
PY
RESP=$(curl -s -F task=invoice -F mode=excel -F "excel=@results/_e2e_inv.xlsx" localhost:8000/api/recognize)
echo "$RESP" | python -m json.tool

echo "==> [4] 导出 Excel"
echo "$RESP" | python -c "import sys,json;d=json.load(sys.stdin);print(json.dumps({'task':d['task'],'results':d['results'],'columns':d['columns']},ensure_ascii=False))" > results/_e2e_export_payload.json
curl -s -X POST -H "Content-Type: application/json" --data @results/_e2e_export_payload.json localhost:8000/api/export -o results/_e2e_export.xlsx
echo "导出文件大小: $(stat -c%s results/_e2e_export.xlsx 2>/dev/null) 字节"
echo "==> 完成"
