# AI 引擎（信息提取）详细设计文档

> 所属系统：出行平台费用报销智能化审核系统 · 能力层（AI 引擎）
> 文档范围：信息提取部分，重点为 **OCR / 票据结构化抽取** 与 **场景识别**，兼及防伪检测与水印解析
> 版本：V1.0　|　日期：2026-06-01　|　状态：评审稿
> 说明：文内模型基准数据来源于公开评测榜单与论文，**数据截至 2026 年 5 月**；该领域模型迭代极快，上线前须以自建测试集复测，详见第 9 章与参考来源。

---

## 目录

1. 模块定位与设计边界
2. 提取任务拆解
3. 模型选型方法论（评测基准）
4. 子模块一：OCR 与票据结构化抽取
5. 子模块二：场景识别
6. 子模块三：防伪检测
7. 子模块四：可信水印解析
8. 推荐技术栈与部署架构
9. 评测与持续迭代
10. 风险与备选
11. 参考来源

---

## 1. 模块定位与设计边界

信息提取是 AI 引擎三大组件（信息提取 → 规则比对 → 审核决策）中的**第一环、也是质量瓶颈**：后续比对与决策的准确率上限，取决于本模块输出字段的准确率与置信度可靠性。

本模块负责将员工上传的照片转换为**结构化、带置信度**的字段数据，输出统一 Schema 供规则比对引擎消费。设计边界如下：

- **纳入**：照片分类路由、OCR 与票据结构化抽取（KIE）、场景识别、防伪检测、水印解析的算法选型与流水线设计。
- **不纳入**：规则比对逻辑、审核决策策略（属下游组件）、基础数据中心建设（属数据层）。
- **硬件前提**：**操作系统 Windows，显卡 NVIDIA RTX A4000（Ampere，16GB GDDR6）**。受 16GB 显存与 Windows 平台约束，推理以 **MinerU 官方部署（vLLM 后端）** 为主；vLLM 在 Windows 上推荐经 **WSL2 / Docker** 运行，亦可退回 Transformers / SGLang 后端；传统视觉模型用 PyTorch + ONNXRuntime/TensorRT。详见第 8 章部署架构。

---

## 2. 提取任务拆解

按照片类型路由到不同提取链路，目标字段与所依赖的子能力如下。

| 照片类型 | 需提取字段 | 依赖子能力 |
|---|---|---|
| 停车水印照片 | 时间、GPS、车架号(VIN)、设备ID | 水印解析（主）+ OCR（校验）+ 防伪 |
| 缴费凭证（停车小票/充电/加油订单/发票） | 金额、交易时间、商户/场站、订单号、地址文本 | OCR + 票据结构化抽取(KIE) + 防伪 |
| 充电桩 / 加油桩照片 | 场景类别、桩编号、关键目标合法性 | 场景识别（检测+分类）+ OCR + 防伪 |
| 无关图 / 不可识别 | — | 照片分类路由（拦截） |

**关键观察**：本场景的票据多为**短文本、键值型**（金额、时间、订单号），不涉及复杂学术公式与多页长文档；因此选型应在「文档解析准确率」之外，**额外侧重票据/小票/印章场景表现、结构化抽取(KIE)能力、单图吞吐与显存占用**。

---

## 3. 模型选型方法论（评测基准）

为使选型可量化、可复现，统一参考以下公开基准，并辅以自建业务测试集（第 9 章）。

- **OmniDocBench V1.6**（端到端文档解析，最新协议）：在 V1.5 基础上修正元素匹配偏差并新增 Hard 困难子集，区分度更高；综合评分覆盖发票、收据、表格、手写、印章等九类文档。截至最新榜单，MinerU2.5-Pro 以 95.69 居 v1.6 **总分**榜首。**但需注意：总分衡量的是复杂整页文档的端到端解析（文本+表格+公式+版面+阅读顺序），与本场景「单张短票据 + 键值抽取(KIE)」所需能力并不一致**，故本模块选型以**收据/印章子项 + KIE 子项**为准，而非总分（详见 4.3）。
- **OCRBench v2 / Nanonets-KIE / Handwritten-KIE**：直接衡量关键信息抽取(KIE)能力，是本场景最相关的基准。
- **COCO mAP@50:95 与 T4 GPU FPS**：目标检测的准确率—速度权衡标准。
- **关键工程指标**：参数量、显存占用、单图/单页吞吐、开源许可证、推理后端兼容性（vLLM/SGLang）。

选型遵循三原则：**准确率（业务子项优先）> 效率（吞吐/显存）> 工程成熟度与许可证友好度**。

---

## 4. 子模块一：OCR 与票据结构化抽取

### 4.1 技术路线判断

2026 年 OCR 已**收敛到紧凑型 VLM（视觉语言模型）路线**：0.9B 级专用模型在文档解析上**全面超越通用大模型**，且推理成本低、可结构化输出。因此本模块主选 VLM 路线，保留传统检测+识别（det+rec）流水线作为高并发简单文本的「快车道」。

### 4.2 候选模型对比

> 评分含 OmniDocBench 综合分与 **KIE 子项**（满分 100，越高越好）；吞吐为官方报告的单卡单并发参考值；实际以复测为准。**本场景应以收据/KIE 子项为主要依据**。

| 模型 | 参数量 | OmniDocBench(总分) | 收据/印章/KIE 子项 | 许可证 | 推理后端 | 备注 |
|---|---|---|---|---|---|---|
| **GLM-OCR**（Z.ai/智谱） | ~0.9B | 94.62（v1.5） | **收据 94.5、印章 90.5；Nanonets-KIE 93.7、Handwritten-KIE 86.1** | MIT（模型）+ Apache（版面） | vLLM/SGLang/Ollama | **原生支持按 JSON Schema 一步直出键值对**；约 2–4GB 显存，吞吐约 1.86 页/s——最贴合本场景 |
| **PaddleOCR-VL-1.5**（百度） | ~0.9B | 94.50（v1.5） | 强；手写略优；KIE 工具链成熟 | Apache 2.0（全开源） | vLLM/Transformers/llama-cpp | 工程完整：版面+识别两阶段、109 语种、PP-StructureV3、**内置 KIE**、DOCX 导出、Docker、多硬件 |
| MinerU2.5-Pro（OpenDataLab/上海AI Lab） | 1.2B | **95.69（v1.6 总分榜首）** | 表格解析 5 项基准居首（PubTabNet 88.4 > GLM-OCR 85.2）；**不直接做 KIE** | 开源，新许可证放宽社区与商用门槛 | vLLM/Transformers/SGLang | 文档解析框架，产物为高保真 Markdown/JSON；做 KIE 需「解析→字段映射」两段式；总分强项（公式/阅读顺序/多页表格）本场景用处有限 |
| dots.ocr-1.5 | 紧凑型 | 高 | 版面+OCR 一体化强 | 开源 | vLLM | 偏「OCR + 更广视觉解析」 |
| PP-OCRv5（传统 det+rec） | 轻量 | 较低（简单文本足够） | — | Apache 2.0 | Paddle/ONNX/浏览器 | SVTR+LCNet，CPU 可跑、极快；作简单纯文本「快车道」与预过滤 |

### 4.3 选型结论

**主选（优先）：GLM-OCR；并列候选：PaddleOCR-VL-1.5；备选：MinerU2.5-Pro（"高保真解析 + 字段映射"路线）**

选型不以 OmniDocBench 总分为唯一依据，而以**本场景的任务形态（单张短票据 + KIE）**为准，理由：

1. **总分 ≠ 本任务**：OmniDocBench 总分衡量复杂整页文档的端到端解析（公式、阅读顺序、多页表格等），这些正是 MinerU2.5-Pro 的强项，但在「单张停车小票/充电加油凭证」上基本用不上；本场景核心是把金额、时间、商户、订单号准确抠出来。
2. **直接 KIE vs 两段式**：MinerU 这类专用解析模型**不直接做关键信息抽取**——其产物是高保真 Markdown/JSON，做 KIE 须再接一步「字段映射」（两段式）。而 **GLM-OCR 原生支持按 JSON Schema 一步直出键值对**，链路更短、更贴合需求。
3. **业务子项更优**：GLM-OCR 收据子项 94.5、印章 90.5，KIE 子项 Nanonets-KIE 93.7、Handwritten-KIE 86.1，均为开源前列，与缴费凭证抽取高度吻合。
4. **工程与许可看 PaddleOCR-VL**：若优先全开源 Apache 2.0 与完整工具链（PP-Structure/内置 KIE/DOCX 导出/Docker/多硬件），PaddleOCR-VL-1.5 是几乎等价的并列候选。
5. **MinerU 何时启用**：当票据高度表格化、或需复用其多格式（PDF/DOCX/PPTX/XLSX）解析能力时，MinerU2.5-Pro 的表格保真优势（如 PubTabNet 88.4 > GLM-OCR 85.2）值得作为「解析+字段映射」路线的备选。

**决定性因素——真实图像分布**：上述分数均基于扫描件/标准文档，而本场景真实输入是**手机拍摄、可能起皱反光、暗光、热敏褪色的小票**，无公开榜单直接衡量。最终主选**须以自建票据测试集做 bake-off 拍板**。

**落地建议**：以 **GLM-OCR 为基线**，在自建票据测试集上与 PaddleOCR-VL-1.5、MinerU2.5-Pro（+字段映射）做 bake-off，按收据字段准确率与单图吞吐定主选；另以 **PP-OCRv5** 作简单纯文本字段的快车道与质量预过滤。三者均为 1B 上下，在 A4000 16GB 上均可运行，对照实验成本低。

### 4.4 票据结构化抽取（KIE）方案

主选 GLM-OCR 时走「**VLM 直出结构化 JSON**」一步；备选 MinerU2.5-Pro 时走「**结构化解析 → 字段映射**」两步。两者最终都产出统一字段。

**路线 A（GLM-OCR 主选，一步直出）**：直接向 OCR-VLM 传入凭证图像 + 抽取指令，要求按目标 Schema 仅输出 JSON。PaddleOCR-VL-1.5 可用其内置 KIE 走类似一步式。

**路线 B（MinerU 备选，两步式）**：
1. MinerU2.5-Pro 将凭证图像解析为高保真 Markdown/JSON（含文本块、表格、阅读顺序）。
2. 字段映射：对短票据用**规则/正则 + 关键字定位**抽取键值；对版式多变者，接一个**轻量 LLM**按目标 Schema 从解析结果中归并字段（输入是结构化文本而非原图，更稳、更省显存）。该路线在票据高度表格化时更有优势。

无论哪条路线，统一目标 Schema：

```text
{
  "amount":    number | null,   // 金额，元
  "pay_time":  string | null,   // ISO8601
  "merchant":  string | null,   // 商户/停车场/油站
  "order_no":  string | null,   // 订单/流水号
  "addr_text": string | null,   // 凭证地址文本
  "field_conf": { "amount": 0~1, "pay_time": 0~1, ... }  // 各字段置信度
}
```

后处理：金额/时间格式归一化、订单号校验、低置信字段降权并触发转人工。对不同商户模板差异，采用「Schema 提示 + 通用兜底」，避免硬编码模板。

---

## 5. 子模块二：场景识别

### 5.1 任务拆解

场景识别需回答三个问题，分别由不同能力承担：

1. **场景属于什么类别？**（充电 / 加油 / 停车 / 无关）→ 图像分类。
2. **关键目标是否存在？**（充电枪、加油机、桩体屏幕、车牌区域）→ 目标检测。
3. **照片是否真实反映该补能行为？**（语义合法性，长尾判断）→ VLM 语义判别。

### 5.2 候选模型对比

#### 5.2.1 目标检测（关键目标定位）

| 模型 | COCO mAP / 速度 | 优势 | 适用 |
|---|---|---|---|
| **RF-DETR**（Roboflow） | RF-DETR-M 约 54.7% mAP @ 约 4.52ms（T4） | 端到端 Transformer，遮挡/复杂场景/域偏移鲁棒，精度领先 | **精度优先**：桩体/加油机/车牌定位 |
| YOLO26（Ultralytics） | 边缘优化，较前代提速可达约 43% | 部署生态成熟、训练简单、边缘友好 | 高吞吐/边缘 |
| YOLOv12 | 引入区域注意力(area attention)/R-ELAN | 速度—精度均衡 | 通用实时 |
| GroundingDINO | 开放词汇、零样本检测 | **无需标注**即可按文本提示检测 | 冷启动期、缺标注数据时 |

#### 5.2.2 场景分类与语义判别

| 模型 | 能力 | 许可证 | 适用 |
|---|---|---|---|
| **SigLIP 2**（Google） | 零样本图像分类，109 语种，NaFlex 动态分辨率 | Apache 2.0 | **零样本场景门控**：给定候选标签即可分类，无需训练 |
| CLIP / MetaCLIP | 零样本分类基线 | 开源 | 备选基线 |
| Qwen3-VL（通用 VLM） | 场景语义判别、长尾问答式判断 | 开源 | 复杂/长尾场景「是否真实在场站补能」 |

### 5.3 选型结论

**组合方案：SigLIP 2（零样本场景门控）+ 微调检测器（RF-DETR 或 YOLO26）+ Qwen3-VL（长尾兜底）**

- **第一道（轻量、快）**：SigLIP 2 零样本分类，对照片打「充电/加油/停车/无关」标签，与申报费用类型做一致性门控；无需标注、即插即用。
- **第二道（精准定位）**：在自建小样本数据上微调 **RF-DETR**（精度优先）或 **YOLO26**（吞吐/边缘优先），检测充电枪、加油机、桩屏、车牌等关键目标，确认场景要素齐全。**冷启动阶段**可先用 GroundingDINO 零样本检测过渡，边收集标注边迭代。
- **第三道（语义兜底）**：对前两道存疑或低置信的样本，调用 **Qwen3-VL** 做语义级判别（如「该图是否真实反映在加油站加油」），处理 OCR/检测难以覆盖的长尾情形。该路较重，仅在边界样本触发，控制成本。

---

## 6. 子模块三：防伪检测

防伪贯穿所有照片类型，识别篡改与造假，是防作弊的第一道技术防线。本场景需覆盖 **PS 局部篡改、翻拍/截图、AIGC 生成/编辑、EXIF 不一致、相册旧图**。

### 6.1 候选与选型

| 能力 | 推荐方案 | 说明 |
|---|---|---|
| 通用篡改定位（IFDL） | **TruFor / MVSS-Net / SparseViT** 类模型 | 输出像素级篡改掩码，识别拼接、复制移动、擦除 |
| 可解释篡改检测 | **FakeShield**（MLLM） | 兼判 PS/DeepFake/AIGC 编辑，输出篡改区域掩码 + 判定依据，适合高风险单据需解释场景 |
| AIGC 生成图鉴别 | 专用 AIGC 分类器（持续学习框架） | 应对新生成器需具备跨模型泛化与持续更新能力，参考 Awesome-AIGC-Image-Video-Detection 收录方法 |
| 翻拍/截图 | 摩尔纹/边框/反光检测（轻量分类） | 屏幕翻拍特征明显，轻量模型即可 |
| EXIF / 元数据一致性 | 规则校验（非 ML） | EXIF 时间/机型/GPS 与水印、OCR 三源交叉 |

### 6.2 策略

- **分层触发**：轻量翻拍/AIGC 分类器对全量照片快速过滤；命中或高金额单据再调用 FakeShield 做可解释复核与篡改定位，控制重模型调用量。
- **三源交叉**：时间与地点分别取自「可信水印 / EXIF / 凭证 OCR」三源，任一显著背离即标记风险（详见水印解析与上游决策）。
- **持续更新**：AIGC 检测须随生成模型演进定期更新权重，纳入第 9 章迭代闭环。

---

## 7. 子模块四：可信水印解析

水印是停车场景中**时间、地点、车架号的权威来源**，可信度高于 OCR 与 EXIF，属提取引擎的高优先级入口。本子模块侧重工程而非大模型选型。

- **嵌入**：客户端拍照 SDK 实时嵌入数字盲水印，payload 含 `时间戳、GPS、VIN、设备指纹、nonce、服务端签名`。
- **解析**：从图像像素域/频域解码 payload → 服务端公钥验签 → 时效校验（拍照到上传 ≤ 24h）→ nonce 查重防重放。
- **失败处理**：解码失败、验签不过或时效异常，一律标记 `watermark_valid=false` 并转人工，**绝不放行**。
- OCR 仅对水印字段做**交叉校验**，不作为唯一来源。

---

## 8. 推荐技术栈与部署架构

### 8.1 推荐技术栈一览

| 子能力 | 主选 | 备选 / 兜底 | 部署 |
|---|---|---|---|
| 照片分类路由 | 轻量分类模型（SigLIP 2 零样本） | 规则 + 元数据 | CPU/GPU |
| OCR + 票据 KIE | **GLM-OCR（直出 KIE）** | PaddleOCR-VL-1.5（内置 KIE）；MinerU2.5-Pro+字段映射；PP-OCRv5（快车道） | vLLM(WSL2/Docker)，A4000 常驻 |
| 场景分类 | **SigLIP 2** | CLIP/MetaCLIP | GPU，毫秒级 |
| 关键目标检测 | **RF-DETR** / YOLO26 | GroundingDINO（冷启动零样本） | PyTorch/ONNX/TensorRT |
| 场景语义兜底 | **小尺寸 Qwen3-VL（2B–8B，量化）** | 远程 VLM 服务/API | GPU，按需触发 |
| 防伪—篡改定位 | TruFor/MVSS-Net；FakeShield（可解释，按需） | — | GPU |
| 防伪—AIGC/翻拍 | 专用分类器 | — | GPU，全量快筛 |
| 水印解析 | 自研解码+验签 | — | CPU + KMS |

### 8.2 部署形态与显存预算（Windows + RTX A4000 16GB）

**平台说明（Windows）**：

- **vLLM 经 WSL2/Docker 运行**：vLLM/SGLang 主要面向 Linux，在 Windows 上建议安装 WSL2（Ubuntu）+ CUDA，并在其中运行 OCR-VLM（GLM-OCR/PaddleOCR-VL，备选 MinerU）的 vLLM 后端；或用官方 Docker 镜像。
- **纯 Windows 退路**：若不便用 WSL2，可走 **Transformers 后端**（原生 Windows + CUDA 可跑，吞吐略低；GLM-OCR 亦支持 Ollama）；传统视觉模型（检测/分类/防伪）用 PyTorch + ONNXRuntime/TensorRT，对 Windows 支持良好。
- A4000 为 Ampere 架构，支持 FP16；显存 16GB，需对模型驻留做规划。

**显存预算（16GB 约束下，关键变化）**：

- **常驻主链路**：OCR-VLM（GLM-OCR ~0.9B，FP16 约 2–4GB）+ SigLIP 2（约 1–2GB）+ 检测器 RF-DETR/YOLO（约 1–3GB）+ 翻拍/AIGC 轻量分类器（< 1GB），合计约 5–10GB，**可在 A4000 上同时常驻**，留出 KV 缓存与图像 token 余量。（备选 MinerU2.5-Pro 为 1.2B，约 2.4–3GB，同样可常驻。）
- **按需触发、避免常驻**：场景语义兜底改用**小尺寸 Qwen3-VL（2B–8B，建议 INT4/INT8 量化）**，FP16 的 7–8B 模型约 14–16GB 会挤占显存，故**量化部署或与主链路错峰调度**；FakeShield 类可解释 MLLM 同样**仅高风险样本触发、用完即卸载**。
- **大模型不本地化**：Qwen3-VL-235B 等大型 VLM 无法在 16GB 本地运行，仅作为远程服务/API 的可选兜底，不进主链路。
- **调度策略**：主链路小模型常驻；重模型（量化 VLM、FakeShield）走**单实例串行 + 显存按需加载/释放**，并控制并发为 1–2，必要时排队，以适配单卡 16GB。

**编排**：照片分类 → 并行调用（OCR / 场景 / 防伪轻量筛）→ 汇聚 → 仅边界样本触发重模型。各子能力独立服务，主链路并发调用压低端到端时延；OCR 用 vLLM 连续批处理，对同一报销单多张照片合并调度。

### 8.3 统一输出 Schema（对接上游比对引擎）

```json
{
  "bill_id": "B202605300001",
  "photos": [{ "photo_id": "P1", "type": "watermark|receipt|device", "oss_url": "..." }],
  "watermark": { "valid": true, "ts": "...", "gps": {"lat":0,"lng":0}, "vin": "...", "device_id": "...", "conf": 0.99 },
  "receipt":   { "amount": 25.00, "pay_time": "...", "merchant": "...", "order_no": "...", "addr_text": "...", "conf": 0.98 },
  "scene":     { "category": "parking|charging|refuel|other", "objects": ["..."], "legal": true, "conf": 0.95 },
  "forgery":   { "tamper": false, "rephoto": false, "aigc": false, "exif_consistent": true, "risk_level": "low|mid|high", "mask_url": null },
  "extracted_at": "...",
  "engine_version": "ocr=GLM-OCR@x; scene=RF-DETR@x; siglip2@x; ..."
}
```

每个字段附带置信度 `conf`，低于阈值的字段在下游决策中降权并倾向转人工。`engine_version` 记录各模型版本，供审计回溯。

---

## 9. 评测与持续迭代

模型基准成绩不能替代业务场景实测，须建立自建评测与回流闭环。

1. **自建测试集**：覆盖停车小票、充电/加油订单、电子发票、桩体照片，含模糊、遮挡、暗光、翻拍、PS、AIGC 等困难样本；按真实分布分层。
2. **核心指标**：字段级准确率（金额/时间/VIN/订单号）、KIE 抽取 F1、场景分类准确率、检测 mAP、防伪召回/误报、端到端单据时延 P95、误放行率。
3. **bake-off**：上线前以 **GLM-OCR 为基线**，对 PaddleOCR-VL-1.5、MinerU2.5-Pro（+字段映射）（及检测器 RF-DETR vs YOLO26）做对照实验，**以收据字段准确率、KIE F1、单图吞吐为主**定主选，而非 OmniDocBench 总分。
4. **回流闭环**：人工复核结论作为标注样本回流，定期微调 OCR/检测/防伪模型与更新 AIGC 检测权重；阈值与模型版本灰度发布、可回滚。

---

## 10. 风险与备选

| 风险 | 影响 | 应对 |
|---|---|---|
| 模型迭代快，榜单数据短期失效 | 选型偏差 | 以自建测试集复测为准；保持可插拔，便于换模 |
| 票据模板差异大、暗光/模糊 | 抽取错误 | Schema 提示 + 通用兜底；低置信转人工并回流 |
| 标注数据不足（场景检测） | 检测精度低 | 冷启动用 GroundingDINO/SigLIP 2 零样本过渡，边收集边微调 |
| 新型 AIGC 造假绕过检测 | 误放行 | 持续更新检测器 + 三源交叉 + 高风险单据 FakeShield 复核 |
| **Windows 下 vLLM 兼容性** | 部署受阻/吞吐下降 | 优先 WSL2/Docker 跑 vLLM；退路用 Transformers 后端；视觉模型走 ONNX/TensorRT |
| **16GB 显存上限** | 重模型挤占、并发受限 | 主链路小模型常驻；量化部署 VLM；重模型按需加载/释放、低并发串行 |
| 唯总分选型导致偏差 | 选错主选 | 以收据/KIE 子项 + 真实样本 bake-off 定主选，不唯 OmniDocBench 总分 |
| MinerU（备选）非直出 KIE | 需额外字段映射 | 主选 GLM-OCR 直出 KIE；若用 MinerU 则接规则/轻量 LLM 做字段映射 |
| 许可证合规 | 商用限制 | GLM-OCR 模型 MIT、版面组件 Apache，使用时同时遵守；PaddleOCR-VL/SigLIP 2 为 Apache 2.0；MinerU2.5-Pro 新许可证已放宽商用 |

---

## 11. 参考来源

> 以下为本文模型数据与结论的主要公开来源，访问于 2026 年 5–6 月。

- OmniDocBench（CVPR 2025 文档解析基准，含 v1.6）：https://github.com/opendatalab/OmniDocBench
- **MinerU / MinerU2.5-Pro（OpenDataLab，GitHub / HF / 论文 arXiv:2604.04771）**：https://github.com/opendatalab/MinerU ；https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B ；https://arxiv.org/pdf/2604.04771
- OmniDocBench V1.5 榜单：https://llm-stats.com/benchmarks/omnidocbench-1.5 ；https://www.codesota.com/ocr/benchmark/omnidocbench
- GLM-OCR（Z.ai/智谱，HuggingFace 模型卡与 GitHub）：https://huggingface.co/zai-org/GLM-OCR ；https://github.com/zai-org/GLM-OCR
- PaddleOCR / PaddleOCR-VL（GitHub 与论文 arXiv:2510.14528）：https://github.com/PaddlePaddle/PaddleOCR ；https://arxiv.org/pdf/2510.14528
- 2026 目标检测模型综述（RF-DETR/YOLO26/YOLOv12）：https://blog.roboflow.com/best-object-detection-models/ ；https://www.ultralytics.com/blog/the-best-object-detection-models-of-2025
- RT-DETR 论文：https://arxiv.org/pdf/2304.08069
- SigLIP 2（Google，零样本分类编码器）：https://huggingface.co/blog/siglip2
- 图像篡改/AIGC 检测：FakeShield（arXiv:2410.02761）；Awesome-AIGC-Image-Video-Detection：https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection

> 注：上述吞吐、显存与基准分数为各来源报告值，受硬件、分辨率、并发等影响，仅供选型参考，实际以自建环境复测为准。
