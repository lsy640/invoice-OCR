# GLM-OCR 票据 / 停车识别 — 前后端一体应用（运行手册）

一个 WebUI + 离线 GLM-OCR 后端，提供两个功能：

1. **发票 / 支付凭证 → 实付金额**（每张图独立）
2. **停车信息**：进出场时间、停车时长、单价(元/小时)、车架号、缴费金额、车牌（一组图=一次停车；电子发票优先以备注的停车时间为准）

输入方式：① 上传图片 ② 图片直链 URL ③ 上传 Excel（图片 URL 为直链）。支持单条/批量、导出 Excel；
**Excel 含原始金额(supply_money/cost)时，识别值与原值不一致的行自动标红提示。**

后端跨平台：**Windows 有 NVIDIA 用 GPU、否则 CPU；macOS（M 系列）用 MLX 加速。**

---

## 一、部署方式速查

| 平台 | 推荐方式 | 命令 |
|---|---|---|
| Windows + NVIDIA | Docker + GPU | 装 Docker Desktop(WSL2) + NVIDIA Container Toolkit，取消 compose 中 `deploy` 注释后 `docker compose up --build` |
| Windows 无 GPU / Linux | Docker(CPU) | `cd OCR/app/docker && docker compose up --build` |
| macOS (M1/M2/M3…) | 原生 + MLX | `bash app/scripts/run_macos_mlx.sh`（**不可用 Docker**，容器访问不到 Metal/MLX） |
| 开发自测(任意) | 本地直跑 | `bash app/scripts/run_local.sh` |

启动后浏览器打开 **http://localhost:8000**。

> 首次启动会联网下载 GLM-OCR 权重(~1.8GB)；之后离线可用。Docker 下权重持久化在命名卷 `glmocr-models`。

---

## 二、Docker 部署（Windows / Linux）

```bash
cd OCR/app/docker
docker compose up --build            # CPU；或启用 GPU 见下
# 打开 http://localhost:8000
```

**启用 NVIDIA GPU**（Windows 需 Docker Desktop + WSL2 + NVIDIA Container Toolkit；Linux 同）：
取消 `docker-compose.yml` 中 `deploy.resources...` 段注释后重新 `up`。容器内 torch 自动判断有无 GPU，
无则回退 CPU（无需改代码）。

镜像构建上下文为仓库根 `OCR/`（compose 已设 `context: ../..`），会打包 `src/ app/ config.yaml`。

---

## 三、macOS 原生 + MLX

```bash
bash app/scripts/run_macos_mlx.sh    # 自动建 venv、装 mlx-vlm、以 MLX 后端启动
```

- 若 `mlx-community` 无现成 GLM-OCR MLX 权重，先转换：
  `python -m mlx_vlm.convert --hf-path zai-org/GLM-OCR -q --mlx-path ./models/GLM-OCR-mlx`，
  并在 `config.yaml` 设 `inference.mlx_repo: "./models/GLM-OCR-mlx"`。
- **若 mlx-vlm 尚不支持 GlmOcr 架构**，后端会自动回退 `transformers`(CPU)；可临时
  `INFERENCE_BACKEND=transformers bash app/scripts/run_local.sh` 走 CPU。

---

## 四、Windows 本地直跑（不喜欢 Docker 时）

PowerShell：
```powershell
cd OCR
python -m venv .venv-app
.\.venv-app\Scripts\pip install -r app\requirements-backend.txt
# 有 NVIDIA 显卡可装 CUDA 版 torch（按 https://pytorch.org 选对应命令），否则默认 CPU
.\.venv-app\Scripts\python app\backend\server.py
```

---

## 五、使用说明

1. 顶部选择功能（发票实付金额 / 停车信息）。
2. 选输入方式并提供数据 → 点「开始识别」→ 结果表格展示。
3. 「导出 Excel」下载结果；不匹配行已标红。

**Excel 列约定**
- 发票：必含 `pic_url`（图片直链）；可选 `supply_money`（原始金额，用于比对）。
- 停车：必含 `cost_images`（**逗号分隔**的图片直链）；可选 `cost`、`vin`（用于比对）。
- 图片直链：若是 `ptmapi.cacxtravel.com/.../obs/download/<id>.jpg` 代理链，后端会**自动替换为公有 OBS 直链**下载（无需 token）。

---

## 六、API（前端即调用以下接口）

| 方法/路径 | 说明 |
|---|---|
| `GET /api/health` | 健康检查 + 当前后端(transformers/mlx) |
| `POST /api/recognize` | multipart：`task=invoice\|parking`、`mode=images\|urls\|excel`、`files[]`/`urls`/`excel` |
| `POST /api/export` | JSON `{task, results, columns}` → 返回高亮 .xlsx |
| `GET /` | 前端页面 |

`curl` 示例：
```bash
curl -F task=invoice -F mode=urls -F 'urls=https://.../a.png' http://localhost:8000/api/recognize
```

---

## 七、配置（`config.yaml`）

```yaml
inference:
  backend: auto        # auto|transformers|mlx（环境变量 INFERENCE_BACKEND 可覆盖）
  device:  auto        # transformers 设备 auto|cuda|cpu（INFERENCE_DEVICE 可覆盖）
  mlx_repo: null       # macOS MLX 权重仓库；null=用 models.glm-ocr.repo
app:
  host: 0.0.0.0
  port: 8000
  max_batch_images: 500
models:
  glm-ocr: { repo: "zai-org/GLM-OCR" }
```

---

## 八、架构

```
前端(静态 HTML/JS)  ──HTTP──>  FastAPI(app/backend/server.py)
                                   ├─ InvoicePipeline(src/invoice_pipeline.py)   只抽实付金额
                                   ├─ ParkingPipeline(src/parking_pipeline.py)   逐图KIE+发票二次OCR+聚合
                                   └─ InferenceBackend(src/inference.py)
                                        ├─ TransformersBackend  (Win NVIDIA→CUDA / CPU; Linux)
                                        └─ MLXBackend           (macOS Apple Silicon)
```
推理逻辑/解析/聚合与命令行批处理(`src/parking_*`)**同一套代码**（`record_from_kie`/`aggregate_group`）。

## 九、排错
- **首次很慢/卡住**：在下载权重(~1.8GB)或 CPU 加载，耐心等；`GET /api/health` 始终可用。
- **GPU 没用上**：确认 NVIDIA 驱动 + Container Toolkit，且 compose 的 `deploy` 段已启用；`docker logs glmocr-app` 看是否 `device=cuda`。
- **transformers 报不识别 glm_ocr**：transformers 必须 ≥5.0（requirements 已约束）。
- **mac 上 MLX 失败**：见第三节，回退 transformers-CPU。
