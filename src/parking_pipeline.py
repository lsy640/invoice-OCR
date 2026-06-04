"""停车识别端到端 pipeline：输入一组图片（URL/本地路径），直接输出完整识别结果。

内部完成：加载图片(ptmapi 代理URL→公有OBS直链) → 逐图 GLM-OCR KIE(水印/支付/发票) →
对未读到停车区间的发票图做备注区域二次OCR补救 → 聚合(进出场时间/时长/单价/车架号/金额/车牌)。

库用法（模型只加载一次，可处理多组）：
    from parking_pipeline import ParkingPipeline
    pipe = ParkingPipeline()
    result = pipe.process([url1, url2, url3], cost=80, vin="LS6...")

命令行（需在 GPU 节点 / conda env `env` 内）：
    python src/parking_pipeline.py --images "url1,url2,url3" [--cost 80] [--vin LS6...] [--out r.json]
    python src/parking_pipeline.py --images-file imgs.txt --cost 80
"""
from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from common import load_config
from inference import InferenceBackend, get_backend
from kie_common import parse_json
from parking_aggregate import aggregate_group
from parking_parse import extract_plate, extract_vin, parse_lease_period, record_from_kie
from prompt import GLM_PARKING_PROMPT

_OCR_TEXT_INSTRUCTION = "Text Recognition:"

_PTM_RE = re.compile(r"/([0-9a-fA-F]{16,}\.(?:jpg|jpeg|png))", re.I)
_REFINE_CROP_TOP = 0.55
_REFINE_UPSCALE = 1600


class ParkingPipeline:
    def __init__(self, backend: InferenceBackend | None = None, config_path: str | None = None):
        self.cfg = load_config(config_path)
        pk = self.cfg["parking"]
        self.obs_base = pk["obs_base"]
        self.max_long_side = int(self.cfg["data"]["max_image_long_side"])
        self.tol = float(self.cfg["evaluation"]["amount_tolerance"])
        self.session = requests.Session()
        self.backend = backend or get_backend(self.cfg)

    # ── 图片加载（直链下载，无需 token）─────────────────────────────
    def _to_obs(self, url: str) -> str:
        m = _PTM_RE.search(url)
        if m and "ptmapi" in url:
            return self.obs_base + m.group(1)
        return url

    def load_image(self, src) -> Image.Image:
        if isinstance(src, Image.Image):
            img = src
        elif isinstance(src, (str, Path)) and Path(src).exists():
            img = Image.open(src)
        else:
            resp = self.session.get(self._to_obs(str(src)), timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")
        if max(img.size) > self.max_long_side:
            s = self.max_long_side / max(img.size)
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
        return img

    def extract_image(self, img: Image.Image) -> dict:
        rec = {"img_type": None, "dt": None, "vin": None, "plate": None, "amount": None,
               "lease_start": None, "lease_end": None, "raw": None, "error": None}
        try:
            raw = self.backend.ocr(img, GLM_PARKING_PROMPT)
            rec["raw"] = raw
            parsed = parse_json(raw)
            if parsed is None:
                rec["error"] = "json_parse_failed"
            else:
                rec.update(record_from_kie(parsed))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"infer_error:{type(e).__name__}:{e}"
        return rec

    def refine_invoice(self, img: Image.Image, rec: dict) -> None:
        """对发票图裁剪备注区域+放大，用全文识别二次 OCR，回填停车区间/车架号/车牌。"""
        w, h = img.size
        crop = img.crop((0, int(h * _REFINE_CROP_TOP), w, h))
        s = _REFINE_UPSCALE / max(crop.size)
        if s > 1:
            crop = crop.resize((round(crop.size[0] * s), round(crop.size[1] * s)), Image.LANCZOS)
        try:
            text = self.backend.ocr(crop, _OCR_TEXT_INSTRUCTION)
        except Exception:  # noqa: BLE001
            return
        rec["refine_raw"] = text
        ls, le = parse_lease_period(text)
        if ls is not None and le is not None:
            rec["lease_start"], rec["lease_end"] = str(ls), str(le)
            rec["img_type"] = "invoice"
        if not rec.get("vin"):
            rec["vin"] = extract_vin(text) or rec.get("vin")
        if not rec.get("plate"):
            rec["plate"] = extract_plate(text) or rec.get("plate")

    # ── 端到端：一组图片 → 完整结果 ────────────────────────────────
    def process(self, images: list, cost: float | None = None, vin: str | None = None) -> dict:
        recs, pil_imgs = [], []
        for src in images:
            try:
                im = self.load_image(src)
                pil_imgs.append(im)
                r = self.extract_image(im)
            except Exception as e:  # noqa: BLE001
                im = None
                r = {"img_type": None, "error": f"load_error:{type(e).__name__}:{e}",
                     "dt": None, "vin": None, "plate": None, "amount": None,
                     "lease_start": None, "lease_end": None, "raw": None}
            r["image"] = str(src)
            recs.append(r)
        # 二次 OCR 补救：发票但未读到停车区间
        for im, r in zip(pil_imgs, recs):
            if im is not None and r.get("img_type") == "invoice" and not (r.get("lease_start") and r.get("lease_end")):
                self.refine_invoice(im, r)
        summary = aggregate_group(recs, cost=cost, gt_vin=vin, tol=self.tol)
        return {"cost": cost, "vin_label": vin, **summary, "images": recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="逗号分隔的图片 URL/本地路径")
    ap.add_argument("--images-file", help="每行一个图片 URL/路径的文件")
    ap.add_argument("--cost", type=float, default=None, help="停车费用标签(可选，用于金额比对)")
    ap.add_argument("--vin", default=None, help="车架号标签(可选，用于车架号比对)")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径(缺省打印到 stdout)")
    args = ap.parse_args()

    if args.images_file:
        images = [l.strip() for l in open(args.images_file, encoding="utf-8") if l.strip()]
    elif args.images:
        images = [x.strip() for x in args.images.split(",") if x.strip()]
    else:
        raise SystemExit("请用 --images 或 --images-file 提供图片")

    pipe = ParkingPipeline()
    result = pipe.process(images, cost=args.cost, vin=args.vin)
    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"已写出 {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
