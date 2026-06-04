"""二次 OCR 补救：对「已判为发票但未解析到停车日期区间」的发票图，裁剪底部备注区域 +
放大，用 GLM-OCR 全文识别模式重新 OCR，再解析停车日期区间/车架号/车牌，回填 pred jsonl。

只处理漏网的发票图（数十张），不重跑全量。完成后重新聚合。

用法（由 run_parking_refine.sh 调用）： python src/parking_refine_invoice.py
"""
from __future__ import annotations

import json
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import load_config, resolve
from parking_download import image_path
from parking_parse import extract_plate, extract_vin, parse_lease_period
from prompt import build_glm_text_messages

CROP_TOP_FRAC = 0.55      # 裁剪底部 45%（覆盖发票备注框的常见位置）
UPSCALE_LONG = 1600       # 裁剪区域放大到长边 ~1600px，提升细字可读性


def _is_invoice_no_lease(r: dict) -> bool:
    raw = r.get("raw") or ""
    has_total = bool(re.search(r'"价税合计":\s*"[^"]+"', raw)) or r.get("img_type") == "invoice"
    no_lease = not (r.get("lease_start") and r.get("lease_end"))
    return has_total and no_lease and not r.get("error")


@torch.inference_mode()
def _ocr_crop(model, processor, img: Image.Image, max_new_tokens: int) -> str:
    w, h = img.size
    crop = img.crop((0, int(h * CROP_TOP_FRAC), w, h))
    scale = UPSCALE_LONG / max(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.size[0] * scale), round(crop.size[1] * scale)), Image.LANCZOS)
    inputs = processor.apply_chat_template(
        build_glm_text_messages(crop), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    cfg = load_config()
    pk = cfg["parking"]
    image_dir = resolve(pk["image_dir"])
    pred_path = resolve(pk["pred_jsonl"])
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]

    targets = [r for r in rows if _is_invoice_no_lease(r)]
    print(f"[refine] 待二次 OCR 的发票图: {len(targets)}", flush=True)
    if not targets:
        return

    model_repo = cfg["models"]["glm-ocr"]["repo"]
    print(f"[refine] 加载模型 {model_repo} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_repo, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    max_new_tokens = int(cfg["inference"]["max_tokens"])

    recovered = 0
    for r in tqdm(targets, desc="refine"):
        p = image_path(image_dir, r["image_id"])
        if not (p.exists() and p.stat().st_size > 0):
            continue
        try:
            text = _ocr_crop(model, processor, Image.open(p).convert("RGB"), max_new_tokens)
        except Exception as e:  # noqa: BLE001
            r["refine_error"] = f"{type(e).__name__}:{e}"
            continue
        r["refine_raw"] = text
        ls, le = parse_lease_period(text)
        if ls is not None and le is not None:
            r["lease_start"], r["lease_end"] = str(ls), str(le)
            r["img_type"] = "invoice"
            recovered += 1
        if not r.get("vin"):
            v = extract_vin(text)
            if v:
                r["vin"] = v
        if not r.get("plate"):
            pl = extract_plate(text)
            if pl:
                r["plate"] = pl

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[refine] 二次 OCR 新解析出停车区间的发票: {recovered}/{len(targets)} -> 已回填 {pred_path}", flush=True)


if __name__ == "__main__":
    main()
