"""任务一·发票/支付凭证：只抽实付金额（不抽补能类型）。

每张图独立识别一个金额；若提供原始金额标签则比对（容差 config.evaluation.amount_tolerance）。
复用推理后端抽象与 kie_common 的 JSON 解析/金额归一化，与停车任务共用同一 backend。
"""
from __future__ import annotations

import re

from PIL import Image

from common import load_config
from inference import InferenceBackend, get_backend
from kie_common import norm_amount, parse_json
from parking_pipeline import ParkingPipeline  # 复用 load_image / _to_obs（URL直链）
from prompt import GLM_AMOUNT_PROMPT


def _amount_from_parsed(parsed: dict) -> float | None:
    """从 KIE JSON 中取金额：优先实付金额，缺则价税合计/含金额的键。"""
    inv_total = pay = None
    for k, v in parsed.items():
        ks = str(k)
        if pay is None and re.search(r"实付|支付|付款金额|实收", ks):
            pay = norm_amount(v)
        if inv_total is None and re.search(r"价税合计|税合计", ks):
            inv_total = norm_amount(v)
    if pay is None and inv_total is None:  # 兜底：任何含「金额」的键
        for k, v in parsed.items():
            if "金额" in str(k):
                pay = norm_amount(v)
                break
    cands = [x for x in (inv_total, pay) if x is not None]
    return max(cands) if cands else None


class InvoicePipeline:
    def __init__(self, backend: InferenceBackend | None = None, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.tol = float(self.cfg["evaluation"]["amount_tolerance"])
        self.backend = backend or get_backend(self.cfg)
        # 复用停车管线的图片加载（含 ptmapi→OBS 直链转换、缩放）；共用同一 backend
        self._loader = ParkingPipeline(backend=self.backend, config_path=config_path)

    def load_image(self, src) -> Image.Image:
        return self._loader.load_image(src)

    def extract_amount(self, img: Image.Image) -> dict:
        out = {"pred_amount": None, "raw": None, "error": None}
        try:
            raw = self.backend.ocr(img, GLM_AMOUNT_PROMPT)
            out["raw"] = raw
            parsed = parse_json(raw)
            if parsed is None:
                out["error"] = "json_parse_failed"
            else:
                out["pred_amount"] = _amount_from_parsed(parsed)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"infer_error:{type(e).__name__}:{e}"
        return out

    def process_one(self, src, label: float | None = None) -> dict:
        """单张图 → {source, pred_amount, label, amount_match, raw, error}。"""
        res = {"source": str(src), "pred_amount": None, "label": label,
               "amount_match": "", "raw": None, "error": None}
        try:
            img = self.load_image(src)
        except Exception as e:  # noqa: BLE001
            res["error"] = f"load_error:{type(e).__name__}:{e}"
            return res
        res.update(self.extract_amount(img))
        if label is not None and res["pred_amount"] is not None:
            res["amount_match"] = "✓" if abs(res["pred_amount"] - float(label)) <= self.tol else "✗"
        return res

    def process(self, images: list, labels: list | None = None) -> list[dict]:
        """一批图片各自独立 → 结果列表。labels 与 images 等长(可含 None)。"""
        labels = labels or [None] * len(images)
        return [self.process_one(s, lb) for s, lb in zip(images, labels)]
