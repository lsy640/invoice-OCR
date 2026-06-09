# GLM-OCR 票据 / 停车识别 — 前后端一体应用（运行手册）

一个 WebUI + 离线 GLM-OCR 后端，提供两个功能：

1. **发票 / 支付凭证 → 实付金额**（每张图独立）
2. **停车信息**：进出场时间、停车时长、单价(元/小时)、车架号、缴费金额、车牌（一组图=一次停车；电子发票优先以备注的停车时间为准）

输入方式：① 上传图片 ② 图片直链 URL ③ 上传 Excel（图片 URL 为直链）。支持单条/批量；
**Excel 含原始金额(supply_money/cost)时，识别值与原值不一致的行自动标红提示。**

后端跨平台：**Windows 有 NVIDIA 用 GPU、否则 CPU；macOS（M 系列）用 MLX 加速（支持 4-bit 量化）；支持 Ollama API 远程调用（无需本地 GPU）。**

---

## 一、部署方式速查


| 平台                    | 推荐方式         | 命令                                                                                                      |
| --------------------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| Windows + NVIDIA      | Docker + GPU | 装 Docker Desktop(WSL2) + NVIDIA Container Toolkit，取消 compose 中 `deploy` 注释后 `docker compose up --build` |
| Windows 无 GPU / Linux | Docker(CPU)  | `cd OCR/app/docker && docker compose up --build`                                                        |
| macOS (M1/M2/M3…)     | 原生 + MLX     | `bash app/scripts/run_macos_mlx.sh`（**不可用 Docker**，容器访问不到 Metal/MLX）                                    |
| 任意平台（无 GPU）           | Ollama API   | 设 `INFERENCE_BACKEND=ollama`，通过远程 Ollama 服务推理（不加载本地模型）                                                |
| 开发自测(任意)              | 本地直跑         | `bash app/scripts/run_local.sh`                                                                         |


启动后浏览器打开 **[http://localhost:8000](http://localhost:8000)**。

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

**MLX 4-bit 量化加速**（推荐）：

```bash
# 将 GLM-OCR 转为 4-bit MLX 量化格式（体积 2.5GB → 1.2GB，推理提速 2-3x）
.venv-mlx/bin/python -m mlx_vlm.convert \
  --hf-path zai-org/GLM-OCR -q --q-bits 4 \
  --mlx-path ./models/GLM-OCR-mlx-4bit --trust-remote-code
```

然后在 `config.yaml` 中设置：

```yaml
inference:
  mlx_repo: "./models/GLM-OCR-mlx-4bit"
```

- **若 mlx-vlm 尚不支持 GlmOcr 架构**，后端会自动回退 `transformers`(CPU)；可临时
`INFERENCE_BACKEND=transformers bash app/scripts/run_local.sh` 走 CPU。

---

## 四、Windows 本地直跑（不喜欢 Docker 时）

PowerShell：

```powershell
cd OCR
# 使用 Python 3.11/3.12 创建虚拟环境（3.14 暂无 CUDA 版 PyTorch 预编译包）
py -3.12 -m venv .venv-app
# 先装 CUDA 版 PyTorch + torchvision（根据驱动支持的 CUDA 版本选择）
.\.venv-app\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 再装其余依赖
.\.venv-app\Scripts\pip install -r app\requirements-backend.txt
# 启动
.\.venv-app\Scripts\python app\backend\server.py
```

> **注意**：`nvidia-smi` 显示的 CUDA Version 是驱动最高支持版本，PyTorch 向下兼容，
> 选 cu121/cu124 均可。直接 `pip install torch` 会装 CPU 版。

验证 GPU 可用：

```powershell
.\.venv-app\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# 应输出 True 12.4
```

---

## 五、Ollama API 远程推理（无需本地 GPU）

适用于本机无 GPU 或希望多客户端共享同一模型服务的场景。需要一台已部署 Ollama + GLM-OCR 的服务器。

### 方式 1：修改配置文件

```yaml
# config.yaml
inference:
  backend: "ollama"
  ollama:
    base_url: "https://ollama.cacxtravel.com"   # Ollama 服务地址
    model: "glm-ocr:latest"                     # 模型名称
    timeout: 120                                # 单次推理超时（秒）
```

### 方式 2：环境变量（优先级更高，适合 Docker）

```bash
# 本地启动
INFERENCE_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.cacxtravel.com python app/backend/server.py

# Docker 启动
docker compose up --build  # 需先在 docker-compose.yml 中取消 OLLAMA 相关环境变量注释
```

### Docker + Ollama（轻量镜像，无需 GPU）

使用 Ollama 后端时，容器不加载本地模型，**无需 GPU、无需下载 1.8GB 权重**，启动秒级。
在 `docker-compose.yml` 中将 `INFERENCE_BACKEND` 改为 `ollama` 并取消 `OLLAMA_BASE_URL` 注释即可。

> Ollama 服务可通过 `https://ollama.cacxtravel.com/api/tags` 查看已部署的模型列表。

---

## 六、性能优化

### 图片分辨率（影响最大）

```yaml
data:
  max_image_long_side: 1280    # 默认值；2048→1280 可提速 2-3x，OCR 精度基本不受影响
```

### SDPA 高效注意力

CUDA 设备自动启用 Scaled Dot-Product Attention（PyTorch 2.0+），无需额外配置。

### bitsandbytes 4-bit 量化（NVIDIA GPU）

```powershell
pip install bitsandbytes
```

```yaml
inference:
  quantize: "4bit"    # 或 "8bit"；null 不量化
```

显存占用从 ~1.8GB 降至 ~0.5GB，自回归生成速度因带宽需求降低而提升。

### torch.compile

```yaml
inference:
  torch_compile: true    # 首次推理慢（编译），后续加速 20-40%
```

### 图片预加载并行化

GPU 推理与图片下载/预处理**流水线并行**，8 线程预加载使 GPU 零等待：

```yaml
app:
  prefetch_workers: 8    # 图片预加载线程数
```

```
优化前（串行）：
[下载img1] → [预处理] → [GPU推理] → [下载img2] → [预处理] → [GPU推理] → ...

优化后（流水线并行）：
线程池: [下载img1] [下载img2] [下载img3] ...
GPU:   [推理img1] → [推理img2] → [推理img3] → ...  ← 图片已预加载就绪
```

### WDDM → TCC 模式（RTX A 系列专业卡）

WDDM 每次 GPU 调用经过 Windows 图形栈，延迟高。若此卡**不接显示器**：

```powershell
# 管理员 PowerShell
nvidia-smi -g 0 -dm 1    # 切 TCC
# 重启电脑；nvidia-smi -g 0 -dm 0 可切回 WDDM
```

---

## 七、使用说明

1. 顶部选择功能（发票实付金额 / 停车信息）。
2. 选输入方式并提供数据 → 点「开始识别」→ 进度条实时显示，结果展示。
3. 各表格均可独立「导出 Excel」下载对应数据；不匹配行已标红。

### 异常数据面板

识别完成后，异常数据会**优先展示在全部结果表上方**，便于快速定位问题：

- **金额不匹配**：当 Excel 提供了标签数据（supply_money / cost）时，识别值与标签不一致的记录在独立红色面板中汇总展示，支持单独导出。
- **车架号不匹配**：当 Excel 提供了 vin 标签时，识别的车架号与标签不一致的记录在独立紫色面板中展示，支持单独导出。
- **停车费异常**（仅停车任务）：计算出的每小时停车单价超过 **25 元/小时** 或低于 **0.1 元/小时** 的记录会在独立的橙色面板中展示，并标注异常原因（单价过高/过低），支持单独导出。
- 页面顶部的**概览条**以标签形式汇总：总条数、匹配数、不匹配数、费用异常数。
- 所有异常面板均可**折叠/展开**，不影响浏览全部数据。

### Excel 列约定

- 发票：必含 `pic_url`（图片直链）；可选 `supply_money`（原始金额，用于比对）。
- 停车：必含 `cost_images`（**逗号分隔**的图片直链）；可选 `cost`、`vin`（用于比对）。
- 图片直链：若是 `ptmapi.cacxtravel.com/.../obs/download/<id>.jpg` 代理链，后端会**自动替换为公有 OBS 直链**下载（无需 token）。

---

## 八、API（前端即调用以下接口）


| 方法/路径                 | 说明                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------- |
| `GET /api/health`     | 健康检查 + 当前后端(transformers/mlx/ollama)                                                |
| `POST /api/recognize` | multipart：`task=invoice|parking`、`mode=images|urls|excel`、`files[]`/`urls`/`excel`；返回 SSE 流（progress→done） |
| `POST /api/export`    | JSON `{task, results, columns}` → 返回高亮 .xlsx（支持自定义列，异常表导出也使用此接口）                   |
| `GET /`               | 前端页面                                                                               |


`curl` 示例：

```bash
curl -F task=invoice -F mode=urls -F 'urls=https://.../a.png' http://localhost:8000/api/recognize
```

---

## 九、配置（`config.yaml`）

```yaml
data:
  max_image_long_side: 1280  # 图片长边上限（影响推理速度，OCR 任务 1280 足够）

inference:
  backend: auto        # auto|transformers|mlx|ollama（环境变量 INFERENCE_BACKEND 可覆盖）
  device:  auto        # transformers 设备 auto|cuda|cpu（INFERENCE_DEVICE 可覆盖）
  quantize: null       # null|"4bit"|"8bit"（CUDA + bitsandbytes）
  torch_compile: false # true 启用编译优化（首次慢，后续快 20-40%）
  mlx_repo: "./models/GLM-OCR-mlx-4bit"  # macOS MLX 4-bit 量化权重路径
  ollama:              # backend 设为 "ollama" 时生效
    base_url: "https://ollama.cacxtravel.com"
    model: "glm-ocr:latest"
    timeout: 120

app:
  host: 0.0.0.0
  port: 8000
  max_batch_images: 3000     # 单次批量上限
  prefetch_workers: 8        # 图片预加载并行线程数

models:
  glm-ocr: { repo: "zai-org/GLM-OCR" }
```

---

## 十、架构

```
前端(静态 HTML/JS)  ──HTTP/SSE──>  FastAPI(app/backend/server.py)
                                       ├─ _prefetch_load()  图片预加载线程池
                                       ├─ InvoicePipeline   只抽实付金额
                                       ├─ ParkingPipeline   逐图KIE+发票二次OCR+聚合
                                       └─ InferenceBackend
                                            ├─ TransformersBackend  (CUDA/CPU, SDPA, 4bit/8bit量化)
                                            ├─ MLXBackend           (macOS Apple Silicon, 4-bit量化)
                                            └─ OllamaBackend        (HTTP API, 远程/本地 Ollama 服务)
```

前端结果展示层级：概览条 → 异常面板(金额不匹配/车架号不匹配/费用异常) → 全部结果表，各层均可独立导出 Excel。

推理逻辑/解析/聚合与命令行批处理(`src/parking_*`)**同一套代码**（`record_from_kie`/`aggregate_group`）。

## 十一、排错

- **首次很慢/卡住**：在下载权重(~1.8GB)或 CPU 加载，耐心等；`GET /api/health` 始终可用。
- **GPU 没用上**：确认 NVIDIA 驱动 + Container Toolkit，且 compose 的 `deploy` 段已启用；日志看是否 `device=cuda`。
- **Windows 下 torch.cuda.is_available() 为 False**：确认用 `--index-url https://download.pytorch.org/whl/cu124` 安装了 CUDA 版 PyTorch，且 Python 版本为 3.11/3.12（3.14 暂无预编译包）。
- **transformers 报不识别 glm_ocr**：transformers 必须 ≥5.0（requirements 已约束）。
- **`torch_dtype` deprecated 警告**：transformers 5.x 已改用 `dtype` 参数名，已适配，可忽略。
- **mac 上 MLX 失败**：见第三节，回退 transformers-CPU。
- **mac 推理慢**：确认已使用 4-bit 量化权重（见第三节），量化后推理速度提升 2-3x。
- **GPU 利用率低**：自回归生成本身 GPU 利用率不高属正常；确认没有其他程序（如 LM Studio）占用显存。
- **Ollama 连接失败**：确认 `base_url` 可访问（浏览器打开 `{base_url}/api/tags` 查看模型列表）；检查网络/VPN。
- **Ollama 推理超时**：调大 `ollama.timeout`（默认 120 秒）；批量图片数多时 Ollama 服务可能排队。
