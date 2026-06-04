"""GLM-OCR 离线抽取（transformers 后端，跑在 conda env `env`：transformers 5.x + torch）。

vLLM 0.19.0 锁定 transformers<5 无法加载 glm_ocr，故 GLM-OCR 走 transformers 直接推理。
对每张票据图，用 GLM-OCR 的 KIE 范式直出 JSON（实付金额 + 补能类型），解析、与标签比对。

特性：单流逐图推理；按 workflow_no 断点续跑；失败隔离；末尾自动评测出 metrics。

用法（由 run_glm_hf.sh 调用）：
  python src/extract_hf.py --model glm-ocr [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

import evaluate
from common import load_config, resolve
from download_images import image_path
from kie_common import fields_from_parsed, parse_json
from prompt import build_glm_messages


def _load_done(pred_path: Path) -> set[str]:
    done: set[str] = set()
    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["workflow_no"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


@torch.inference_mode()
def _generate(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    messages = build_glm_messages(image)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def run(model_name: str, model_repo: str, limit: int | None = None) -> Path:
    cfg = load_config()
    image_dir = resolve(cfg["data"]["image_dir"])
    results_dir = resolve(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / f"{model_name}_pred.jsonl"
    max_new_tokens = int(cfg["inference"]["max_tokens"])

    with open(resolve(cfg["data"]["records_json"]), "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit is not None:
        records = records[:limit]

    done = _load_done(pred_path)
    todo = [r for r in records if r["workflow_no"] not in done]
    print(f"[{model_name}] 总 {len(records)} 条，已完成 {len(done)} 条，待处理 {len(todo)} 条", flush=True)

    print(f"[{model_name}] 加载模型 {model_repo} ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_repo, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()

    with open(pred_path, "a", encoding="utf-8") as fout:
        for rec in tqdm(todo, desc=model_name):
            wf = rec["workflow_no"]
            out: dict = {
                "workflow_no": wf, "gt_amount": rec.get("gt_amount"), "gt_type": rec.get("gt_type"),
                "pred_amount": None, "pred_type": None, "raw": None, "error": None,
            }
            img_p = image_path(image_dir, wf, rec["pic_url"])
            if not (img_p.exists() and img_p.stat().st_size > 0):
                out["error"] = "missing_image"
            else:
                try:
                    image = Image.open(img_p).convert("RGB")
                    raw = _generate(model, processor, image, max_new_tokens)
                    out["raw"] = raw
                    parsed = parse_json(raw)
                    if parsed is None:
                        out["error"] = "json_parse_failed"
                    else:
                        out["pred_amount"], out["pred_type"] = fields_from_parsed(parsed)
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"infer_error:{type(e).__name__}:{e}"
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[{model_name}] 抽取完成 -> {pred_path}", flush=True)
    return pred_path


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-ocr", choices=list(cfg["models"].keys()))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model_repo = cfg["models"][args.model]["repo"]
    pred_path = run(args.model, model_repo, args.limit)

    tol = float(cfg["evaluation"]["amount_tolerance"])
    m = evaluate.evaluate_model(pred_path, tol)
    out = resolve(cfg["paths"]["results_dir"]) / f"metrics_{args.model}.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.model}] 指标: {json.dumps(m, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
