# GLM-OCR 票据 / 停车识别

用 GLM-OCR 离线做两类关键信息抽取(KIE)，提供两种用法：
**① Web 应用**（交付最终用户在 Windows/macOS 本地离线运行）；**② 批处理评测**（在 GPU 集群复现准确率）。两者共用同一套抽取/解析/聚合代码。

- **任务一·发票/支付凭证** → 实付金额
- **任务二·停车数据** → 进出场时间 / 停车时长 / 单价(元/小时) / 车架号 / 缴费金额 / 车牌
  （一组图=一次停车；电子发票优先以备注的停车时间为准）

> 任务一在 **Web 应用中仅抽实付金额**；批处理评测版还含补能类型(加油/充电)以算准确率（历史数据）。

---

## 一、Web 应用（前后端一体，推荐）

WebUI + FastAPI 后端，本地离线运行。用户选功能 → 三种输入（上传图片 / 图片直链 URL / 上传 Excel）→ 单条或批量识别 → 导出 Excel；**Excel 含原始金额(`supply_money`/`cost`)时，识别值与原值不一致的行自动标红提示。**

**跨平台后端**（`src/inference.py` 抽象，自动选择）：Windows 有 NVIDIA→GPU、否则 CPU；macOS(M 系列)→MLX。

### 部署速查（详细运行手册见 **[app/README_app.md](app/README_app.md)**）

| 平台 | 方式 | 命令 |
|---|---|---|
| Windows+NVIDIA / 无GPU / Linux | Docker | `cd app/docker && docker compose up --build`（GPU：装 NVIDIA Container Toolkit + 取消 compose `deploy` 注释）|
| macOS (Apple Silicon) | 原生 + MLX | `bash app/scripts/run_macos_mlx.sh`（Docker 访问不到 Metal/MLX，故原生）|
| 任意（开发自测） | 本地直跑 | `bash app/scripts/run_local.sh` |

启动后打开 **http://localhost:8000**。首次联网下载 GLM-OCR 权重(~1.8GB)，之后离线可用。

### Excel 列约定
- **发票**：必含 `pic_url`（图片直链）；可选 `supply_money`（原始金额，用于比对）。
- **停车**：必含 `cost_images`（**逗号分隔**直链）；可选 `cost`、`vin`。
- 直链：`ptmapi.cacxtravel.com/.../obs/download/<id>` 代理链后端**自动转公有 OBS 直链**（无需 token）。

### 架构 / API
```
前端(静态HTML/JS) ──HTTP──> FastAPI(app/backend/server.py)
                              ├─ InvoicePipeline(src/invoice_pipeline.py)  仅实付金额
                              ├─ ParkingPipeline(src/parking_pipeline.py)  逐图KIE+发票二次OCR+聚合
                              └─ InferenceBackend(src/inference.py)
                                   ├─ TransformersBackend (Win NVIDIA→CUDA / CPU; Linux)
                                   └─ MLXBackend          (macOS Apple Silicon)
```
推理/解析/聚合与批处理**同一套代码**（`record_from_kie` / `aggregate_group`）。
API：`POST /api/recognize`(multipart: `task`/`mode`/`files`|`urls`|`excel`)、`POST /api/export`(→高亮 xlsx)、`GET /api/health`。
应用配置在 `config.yaml` 的 `inference`(backend/device/mlx_repo) 与 `app`(host/port) 段。

---

## 二、批处理评测（GPU 集群复现准确率）

下述为在 conda env `env`（transformers 5.x + torch）GLM-OCR 离线推理的批处理评测，与应用共用抽取/聚合代码；图片走公有 OBS 直链（无需 token）。

## 任务一：发票/支付凭证（实付金额 + 补能类型）

**当前仅抽两个字段**：实付金额（amount）、补能类型（加油=0 / 充电=1），与
`data/发票数据.xlsx` 的 `supply_money` / `supplementation_type` 标签比对算准确率。

## 当前结果（GLM-OCR，全量 1000）

| 指标 | 数值 |
|---|---|
| 解析失败率 | **0.0%** |
| 金额准确率 | **94.2%** |
| 类型准确率 | **86.3%** |
| 联合准确率（两者皆对） | **82.4%** |

详见 [`results/REPORT_glm-ocr.md`](results/REPORT_glm-ocr.md)；逐条对比见
[`results/识别结果对比_glm-ocr.xlsx`](results/识别结果对比_glm-ocr.xlsx)。

## 架构与运行时

- **模型**：GLM-OCR（`zai-org/GLM-OCR`，~0.9B），用其 KIE 范式直出 JSON（设计文档路线A）。
- **运行时**：conda env **`env`（transformers 5.4.0 + torch cu126）的 transformers 离线推理**。
  - 原因：GLM-OCR 的 `config.json` 要求 transformers 5.x，而 vLLM 0.19.0 锁定 `transformers<5`
    无法加载 `glm_ocr`；故绕开 vLLM，亦**不影响 env_vllm**（你的 VLM 项目用）。
- **全部在计算节点执行**：计算节点有外网，依赖安装、数据规范化、图片下载、模型权重下载都在
  GPU 作业内完成（无需登录节点预处理）。模型权重由 transformers 自动从 HF 拉取并缓存到
  `config.yaml` 的 `paths.hf_home`（NFS，作业间复用）。
- **集群约定**：Slurm，分区 `MGPU-TC2`，固定 `TC2N08`，`normal` QOS 上限 6h/30G；
  Slurm 在 spool 目录执行脚本，故作业脚本用**绝对项目路径**（不能用 `BASH_SOURCE`）。

## 目录

```
config.yaml          # 数据/模型/推理/应用 集中配置
requirements.txt     # 数据层依赖（pandas/openpyxl/pillow/requests/...）
src/
  common.py          # 配置加载与路径解析（基于 __file__，与 cwd 无关）
  inference.py       # 推理后端抽象：TransformersBackend(GPU/CPU) / MLXBackend(macOS) / get_backend
  prompt.py          # GLM-OCR KIE prompt（发票金额 / 停车统一槽 / 全文识别）
  kie_common.py      # JSON 解析 + 金额/类型归一化 + 字段提取（物理单位优先判类型）
  # —— 任务一·发票（批处理算准确率）——
  prepare_data.py    # xlsx -> data/records.json
  download_images.py # 下载图片到 data/images/（缓存、续跑、超大图缩放）
  extract_hf.py      # GLM-OCR transformers 抽取 -> results/glm-ocr_pred.jsonl + 评测
  evaluate.py / rescore.py / export_excel.py   # 评测 / 重算 / 合并导出(✓✗)
  invoice_pipeline.py# 【应用用】只抽实付金额（接 inference 后端）
  # —— 任务二·停车 ——
  parking_prepare.py / parking_download.py     # xlsx->records / OBS直链下载
  parking_extract_hf.py / parking_aggregate.py # 逐图抽取(批) / 按行聚合(aggregate_group)
  parking_parse.py   # 日期区间/车架号/车牌解析 + record_from_kie
  parking_refine_invoice.py  # 发票备注区域二次OCR补救
  parking_pipeline.py# 【应用用】端到端 ParkingPipeline（接 inference 后端）
scripts/             # 批处理 sbatch（Slurm/TC2N08）
  prepare.sh         # 准备片段（依赖+规范化+下载），被作业脚本 source
  run_glm_hf.sh      # 发票全量抽取+评测
  run_parking_glm.sh / run_parking_refine.sh / run_parking_pipeline.sh  # 停车批处理/补救/端到端
app/                 # 前后端一体 Web 应用（见 app/README_app.md）
results/             # 预测 jsonl、指标 json、报告、对比 Excel
```

## 使用

```bash
# 1) GPU 节点：GLM-OCR 全量抽取 + 评测（单作业，约 10 分钟）
sbatch OCR/scripts/run_glm_hf.sh
squeue -la
#   作业内：装数据依赖 → 规范化1000条 → 下载图片 → transformers 加载 GLM-OCR → 逐图 KIE → 评测
#   超时前 ssh 登录节点自我重提交、断点续跑（按 workflow_no 跳过 *_pred.jsonl 已完成项）。
#   小样本调试： sbatch --export=ALL,LIMIT=20 OCR/scripts/run_glm_hf.sh

# 2) 看结果
cat OCR/results/metrics_glm-ocr.json
cat OCR/results/REPORT_glm-ocr.md

# 3) 导出合并对比 Excel（原始数据 + 预测 + amount_match/type_match/all_match）
/home/msai/lius0131/.conda/envs/env/bin/python OCR/src/export_excel.py --model glm-ocr

# 4) 仅按更新后的解析逻辑重算指标（不重跑推理，利用已存的 raw）
/home/msai/lius0131/.conda/envs/env/bin/python OCR/src/rescore.py --model glm-ocr
```

## 产物说明

- `results/glm-ocr_pred.jsonl`：逐条 `workflow_no / 标签 / pred_amount / pred_type / raw / error`。
- `results/metrics_glm-ocr.json`：金额/类型/联合准确率、解析失败率、样本数。
- `results/识别结果对比_glm-ocr.xlsx`：原始表全部列 + `pred_amount`、`pred_type(_text)`、
  `amount_match`、`type_match`、`all_match`(✓/✗)、`raw_output`、`error`。
- `results/REPORT_glm-ocr.md`：方法、结果（含 v1→v2 优化对比）、误差分析。

---

## 任务二：停车数据（水印 + 支付凭证 → 时长/单价）

`data/停车数据.xlsx`：`vin`(车架号)、`cost`(停车费用)、`cost_images`(逗号分隔的多张图 URL)。
每行含若干张图，分三类：
- **进出场水印照片**：左下角水印含 日期+时间、完整车架号(17位)、有时含车牌（今日水印/宽凳等多种版式）。
- **支付截图/缴费凭证**：含 缴费金额、缴费时间、有时含车牌。
- **电子发票**（普通发票）：备注栏含「停车时间（…-…）」或「租赁期起止：…」+ 车架号/车牌，价税合计为总额。

**图片直链**：`cost_images` 是 ptmapi 代理 URL（需 JWT 登录），但对象在公有 OBS 桶，
**只需把前缀 `https://ptmapi.cacxtravel.com/thirdparty/huaweiyun/obs/download/` 换成
`https://obs-prod-biz.obs.cn-east-3.myhuaweicloud.com/changan/`** 即可无 token 直链下载。

**流程**：逐图 GLM-OCR 抽 `日期/时间/完整车架号/车牌号/支付金额/价税合计/停车开始时间/停车结束时间`
→ 后处理判图片类型(水印/支付/invoice) → 按行聚合：
- **若行内有电子发票（备注解析出停车起止时间），则进出场时间以发票为准**（不再用水印推算），
  缴费金额取发票价税合计；
- 否则进出场时间 = 非支付照片时间的 min=进场、max=出场，缴费金额取支付凭证最大值（多张多为重复）；
- 停车时长 = 出场 − 进场；单价 = cost / 时长；车架号取众数并与 `vin` 比对，金额与 `cost` 比对。

```bash
sbatch OCR/scripts/run_parking_glm.sh           # 全量（798行/~2687图，约35分钟）
#   小样本： sbatch --export=ALL,LIMIT=5 OCR/scripts/run_parking_glm.sh
sbatch OCR/scripts/run_parking_refine.sh        # 二次OCR补救（仅漏网发票，约2分钟）+ 重新聚合
#   仅重算聚合(不重跑推理)： python OCR/src/parking_aggregate.py
```

**二次 OCR 补救**（`parking_refine_invoice.py`）：对「已判为发票但备注没解析出停车日期区间」的发票图，
裁剪底部备注区域 + 放大，用 GLM-OCR 全文识别模式(`Text Recognition:`)重新 OCR，再解析日期区间/
车架号/车牌回填——比 KIE 槽填更擅长读细字。本数据集补回 18/39 张（其余多为「租赁期同一天」的
单日停车或图像过糊，合理回退水印时间）。

**产物** `results/停车识别结果.xlsx`：
- `汇总` sheet：原始列 + `time_source`(invoice/watermark)、`num_invoice`、
  `entry_time`/`exit_time`/`park_hours`/`unit_price_yuan_per_h`、
  `vin_detected`/`vin_match`、`payment_amount`/`amount_match`、`payment_time`/`plate`。
- `逐图明细` sheet：每张图的 `img_type`/`dt`/`vin`/`plate`/`amount`/`lease_start`/`lease_end`/`raw`。

**当前结果（798 行）**：
- 缴费金额匹配 **87.5%**（698/798）；其中**电子发票行金额匹配 95/97**。
- 车架号匹配 **85.6%**（683/798）；可算单价 **790/798**（中位 1.25 元/h）。
- 含电子发票行 **133**；其中 `time_source=invoice`（备注解析出停车日期区间、用作权威停车时间）**115**
  （含二次 OCR 补回的 18 张），其余为单日租赁(0时长)或图像过糊则回退水印时间（金额仍用价税合计）。
- 电子发票识别：**读到「价税合计」即判为发票**；用「备注」槽整段 OCR 备注框，再鲁棒解析停车
  日期区间（兼容「停车时间（…-…）」「租赁期起止：…」、裸日期范围、结束日期缺年份等多种写法）。

相关文件：`src/parking_prepare.py`、`parking_download.py`、`parking_parse.py`、
`parking_extract_hf.py`、`parking_aggregate.py`、`parking_refine_invoice.py`、`scripts/run_parking_glm.sh`。

### 端到端 pipeline（输入一组图片 → 直接输出完整结果）

`src/parking_pipeline.py` 把全流程封装为单一入口：**加载图片(ptmapi代理URL自动转OBS直链) →
逐图 GLM-OCR(水印/支付/发票) → 发票备注二次OCR补救 → 聚合**，直接返回完整识别结果。
逐图解析(`record_from_kie`)与按组聚合(`aggregate_group`)与批处理同一套口径（单一真源）。

```bash
# GPU 节点：对一组图片直接出结果 JSON
sbatch --export=ALL,IMAGES_FILE=imgs.txt,COST=80,VIN=LS6... OCR/scripts/run_parking_pipeline.sh
#   或直接传 URL（分号分隔以避开 --export 逗号）：
sbatch "--export=ALL,IMAGES=url1;url2;url3,COST=80" OCR/scripts/run_parking_pipeline.sh
#   产物：results/pipeline_result_<jobid>.json
```

库用法（模型只加载一次，可连续处理多组）：
```python
from parking_pipeline import ParkingPipeline
pipe = ParkingPipeline()
result = pipe.process([url1, url2, url3], cost=80, vin="LS6...")
# result: {time_source, entry_time, exit_time, park_hours, unit_price_yuan_per_h,
#          vin_detected, vin_match, payment_amount, amount_match, plate, images:[逐图...]}
```

## 备注

- **断点续跑**：抽取按 `workflow_no` 跳过 `*_pred.jsonl` 已完成项；作业被打断重新 `sbatch` 即接着跑。
- **改参数**：模型仓库、金额容差、推理后端等都在 `config.yaml`。
- **PaddleOCR-VL**：早期 bake-off 实测其为纯 OCR/版面解析模型、不一步直出 JSON，故未采用；
  相关 vLLM 脚本已精简删除（如需可从 git 历史恢复）。当前两任务均用 GLM-OCR。
