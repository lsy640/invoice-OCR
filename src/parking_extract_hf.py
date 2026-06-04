"""停车图片逐图抽取（GLM-OCR，transformers 后端，conda env `env`）。

对每行的每张图，用 GLM-OCR 抽取 {日期时间, 完整车架号, 车牌号, 支付金额}，
解析为 datetime/vin/plate/amount 并判定图片类型，写 parking_glm_pred.jsonl（按 行+图id 续跑）。

用法（由 run_parking_glm.sh 调用）： python src/parking_extract_hf.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import load_config, resolve
from kie_common import parse_json
from parking_download import image_path
from parking_parse import record_from_kie
from prompt import build_glm_parking_messages


def _load_done(pred_path: Path) -> set[str]:
    done: set[str] = set()
    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add(f"{r['workflow_no']}|{r['image_id']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


@torch.inference_mode()
def _generate(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    messages = build_glm_parking_messages(image)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run(limit: int | None = None, workflows: set[str] | None = None) -> Path:
    cfg = load_config()
    pk = cfg["parking"]
    image_dir = resolve(pk["image_dir"])
    pred_path = resolve(pk["pred_jsonl"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(cfg["inference"]["max_tokens"])
    model_repo = cfg["models"]["glm-ocr"]["repo"]

    with open(resolve(pk["records_json"]), "r", encoding="utf-8") as f:
        records = json.load(f)
    if workflows:
        records = [r for r in records if r["workflow_no"] in workflows]
    if limit is not None:
        records = records[:limit]

    # 展开成 (record, image) 任务
    tasks = [(r, im) for r in records for im in r["images"]]
    done = _load_done(pred_path)
    todo = [(r, im) for r, im in tasks if f"{r['workflow_no']}|{im['image_id']}" not in done]
    print(f"[parking] {len(records)} 行 / {len(tasks)} 图，已完成 {len(done)}，待处理 {len(todo)}", flush=True)

    print(f"[parking] 加载模型 {model_repo} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_repo, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()

    with open(pred_path, "a", encoding="utf-8") as fout:
        for rec, im in tqdm(todo, desc="parking"):
            oid = im["image_id"]
            out: dict = {
                "workflow_no": rec["workflow_no"], "image_id": oid,
                "img_type": None, "dt": None, "vin": None, "plate": None, "amount": None,
                "lease_start": None, "lease_end": None,
                "raw": None, "error": None,
            }
            p = image_path(image_dir, oid)
            if not (p.exists() and p.stat().st_size > 0):
                out["error"] = "missing_image"
            else:
                try:
                    raw = _generate(model, processor, Image.open(p).convert("RGB"), max_new_tokens)
                    out["raw"] = raw
                    parsed = parse_json(raw)
                    if parsed is None:
                        out["error"] = "json_parse_failed"
                    else:
                        out.update(record_from_kie(parsed))
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"infer_error:{type(e).__name__}:{e}"
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[parking] 抽取完成 -> {pred_path}", flush=True)
    return pred_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 行")
    ap.add_argument("--workflows", default=None, help="逗号分隔的 workflow_no，仅处理这些行（调试用）")
    args = ap.parse_args()
    wfs = set(re.split(r"[,;]", args.workflows)) if args.workflows else None
    run(args.limit, wfs)


if __name__ == "__main__":
    main()
