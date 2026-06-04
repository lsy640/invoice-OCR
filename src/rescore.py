"""离线重评分：用更新后的解析逻辑重新解析已存的 raw 输出，重算字段与指标。

不重跑模型推理，仅对 results/<model>_pred.jsonl 中的 raw 重新抽取 amount/type，
覆盖写回 pred jsonl 并重算 metrics_<model>.json。

用法：
  python src/rescore.py --model glm-ocr
"""
from __future__ import annotations

import argparse
import json

import evaluate
from common import load_config, resolve
from kie_common import fields_from_parsed, parse_json


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-ocr", choices=list(cfg["models"].keys()))
    args = ap.parse_args()

    results_dir = resolve(cfg["paths"]["results_dir"])
    pred_path = results_dir / f"{args.model}_pred.jsonl"
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]

    changed = 0
    for r in rows:
        if r.get("error"):
            continue
        parsed = parse_json(r.get("raw") or "")
        if parsed is None:
            continue
        amt, typ = fields_from_parsed(parsed)
        if amt != r.get("pred_amount") or typ != r.get("pred_type"):
            changed += 1
        r["pred_amount"], r["pred_type"] = amt, typ

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tol = float(cfg["evaluation"]["amount_tolerance"])
    m = evaluate.evaluate_model(pred_path, tol)
    (results_dir / f"metrics_{args.model}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"重评分完成：更新 {changed} 条 / 共 {len(rows)} 条")
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
