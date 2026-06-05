"""推理后端抽象：统一接口 ocr(image, instruction) -> str，可切换 transformers / MLX。

- TransformersBackend：Windows 有 NVIDIA 用 CUDA，否则 CPU；Linux 同理。
- MLXBackend：macOS（Apple M 系列）用 mlx-vlm 加速（尽力实现）。
- get_backend(cfg)：按 config/env/平台自动选择，MLX 失败自动回退 transformers。

供 InvoicePipeline / ParkingPipeline 共用同一个后端实例（模型只加载一次）。
"""
from __future__ import annotations

import os
import platform
from abc import ABC, abstractmethod

from PIL import Image


class InferenceBackend(ABC):
    name = "base"

    @abstractmethod
    def ocr(self, image: Image.Image, instruction: str) -> str:
        """给定图片 + 指令文本，返回模型输出文本。"""
        raise NotImplementedError


class TransformersBackend(InferenceBackend):
    name = "transformers"

    def __init__(self, repo: str, device: str | None = None, max_new_tokens: int = 256,
                 use_compile: bool = False, quantize: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        if device in (None, "", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # CUDA 优化：启用 SDPA（Scaled Dot-Product Attention）高效注意力
        attn_impl = "sdpa" if device == "cuda" else None

        # 量化配置：4bit/8bit（仅 CUDA，需 bitsandbytes）
        quant_config = None
        if quantize and device == "cuda":
            from transformers import BitsAndBytesConfig
            if quantize == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantize == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"[backend:transformers] 量化模式: {quantize}", flush=True)

        print(f"[backend:transformers] 加载 {repo} (device={device}, dtype={dtype}, attn={attn_impl}) ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        model_kwargs = dict(
            dtype=dtype,
            device_map=("cuda" if device == "cuda" else None),
            trust_remote_code=True,
        )
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        self.model = AutoModelForImageTextToText.from_pretrained(repo, **model_kwargs)
        if device != "cuda" and not quant_config:
            self.model = self.model.to(device)
        self.model.eval()

        # torch.compile：首次推理慢（编译），后续加速 20-40%
        if use_compile and device == "cuda" and not quant_config:
            print("[backend:transformers] torch.compile 编译中（首次推理会较慢）...", flush=True)
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def ocr(self, image: Image.Image, instruction: str) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}],
        }]
        with self.torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class MLXBackend(InferenceBackend):
    """macOS Apple Silicon 用 mlx-vlm 跑 GLM-OCR（尽力实现，需在 mac 上验证）。

    mlx-vlm 各版本 generate/apply_chat_template 签名略有差异，这里做了防御性适配。
    若该机型/版本不支持 GlmOcr 架构，get_backend 会自动回退 transformers。
    """
    name = "mlx"

    def __init__(self, repo: str, max_new_tokens: int = 256):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config as mlx_load_config

        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self.max_new_tokens = max_new_tokens
        print(f"[backend:mlx] 加载 {repo} ...", flush=True)
        self.model, self.processor = load(repo, trust_remote_code=True)
        try:
            self.mlx_cfg = mlx_load_config(repo)
        except Exception:  # noqa: BLE001
            self.mlx_cfg = {}

    def ocr(self, image: Image.Image, instruction: str) -> str:
        try:
            prompt = self._apply_chat_template(self.processor, self.mlx_cfg, instruction, num_images=1)
        except TypeError:
            prompt = self._apply_chat_template(self.processor, self.mlx_cfg, instruction)
        try:
            out = self._generate(self.model, self.processor, prompt, image=[image],
                                 max_tokens=self.max_new_tokens, verbose=False)
        except TypeError:
            out = self._generate(self.model, self.processor, image, prompt,
                                 max_tokens=self.max_new_tokens, verbose=False)
        if isinstance(out, str):
            return out
        return getattr(out, "text", None) or (out[0] if isinstance(out, (tuple, list)) else str(out))


def _mlx_available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def get_backend(cfg: dict, override: str | None = None) -> InferenceBackend:
    """按 config / 环境变量 / 平台自动选择推理后端。

    选择优先级：override > 环境变量 INFERENCE_BACKEND > config.inference.backend。
    auto = Darwin+arm64+mlx可用 → mlx；否则 transformers（CUDA 可用则 GPU，否则 CPU）。
    """
    inf = cfg.get("inference", {})
    repo = cfg["models"]["glm-ocr"]["repo"]
    mlx_repo = inf.get("mlx_repo") or repo
    max_new = int(inf.get("max_tokens", 256))
    backend = (override or os.environ.get("INFERENCE_BACKEND") or inf.get("backend", "auto")).lower()
    device = os.environ.get("INFERENCE_DEVICE") or inf.get("device", "auto")

    if backend == "auto":
        is_mac_arm = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
        backend = "mlx" if (is_mac_arm and _mlx_available()) else "transformers"

    if backend == "mlx":
        try:
            return MLXBackend(mlx_repo, max_new_tokens=max_new)
        except Exception as e:  # noqa: BLE001
            print(f"[backend] MLX 初始化失败({type(e).__name__}: {e})，回退 transformers", flush=True)

    use_compile = str(inf.get("torch_compile", False)).lower() in ("true", "1", "yes")
    quantize = inf.get("quantize")  # null / "4bit" / "8bit"
    return TransformersBackend(repo, device=device, max_new_tokens=max_new,
                               use_compile=use_compile, quantize=quantize)
