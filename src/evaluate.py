"""评测：把预测 JSONL 与标签比对，产出单模型指标和双模型对比报告。

指标：
- amount_acc  金额准确率：|pred-gt| <= 容差 视为命中（仅在 gt_amount 存在的样本上算）
- type_acc    类型准确率：pred_type == gt_type
- joint_acc   联合准确率：金额与类型同时命中
- parse_fail_rate  解析/请求失败率
- n_total / n_evaluated

用法：
  python src/evaluate.py                       # 评测所有已存在的 *_pred.jsonl
  python src/evaluate.py --model glm-ocr       # 仅评测单个模型
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, resolve


def _load_preds(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_model(pred_path: Path, tol: float) -> dict:
    rows = _load_preds(pred_path)
    n_total = len(rows)
    n_error = sum(1 for r in rows if r.get("error"))

    amount_hits = amount_total = 0
    type_hits = type_total = 0
    joint_hits = joint_total = 0

    for r in rows:
        gt_amount, gt_type = r.get("gt_amount"), r.get("gt_type")
        pred_amount, pred_type = r.get("pred_amount"), r.get("pred_type")

        amt_ok = type_ok = None
        if gt_amount is not None:
            amount_total += 1
            amt_ok = pred_amount is not None and abs(pred_amount - gt_amount) <= tol
            amount_hits += int(amt_ok)
        if gt_type is not None:
            type_total += 1
            type_ok = pred_type == gt_type
            type_hits += int(type_ok)
        if gt_amount is not None and gt_type is not None:
            joint_total += 1
            joint_hits += int(bool(amt_ok) and bool(type_ok))

    def _acc(h, t):
        return round(h / t, 4) if t else None

    return {
        "pred_file": pred_path.name,
        "n_total": n_total,
        "n_error": n_error,
        "parse_fail_rate": round(n_error / n_total, 4) if n_total else None,
        "amount_acc": _acc(amount_hits, amount_total),
        "amount_evaluated": amount_total,
        "type_acc": _acc(type_hits, type_total),
        "type_evaluated": type_total,
        "joint_acc": _acc(joint_hits, joint_total),
        "joint_evaluated": joint_total,
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.2f}%" if isinstance(v, float) and v <= 1 else str(v)


def write_comparison(metrics: dict[str, dict], out_path: Path) -> None:
    lines = ["# GLM-OCR vs PaddleOCR-VL-1.5 抽取准确率对比", ""]
    lines.append("| 指标 | " + " | ".join(metrics.keys()) + " |")
    lines.append("|---|" + "---|" * len(metrics))
    rows = [
        ("样本数", "n_total"),
        ("金额准确率", "amount_acc"),
        ("类型准确率", "type_acc"),
        ("联合准确率", "joint_acc"),
        ("解析失败率", "parse_fail_rate"),
        ("失败条数", "n_error"),
    ]
    for label, key in rows:
        cells = [_fmt(metrics[m].get(key)) for m in metrics]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="仅评测指定模型；缺省评测全部已存在预测")
    args = ap.parse_args()

    results_dir = resolve(cfg["paths"]["results_dir"])
    tol = float(cfg["evaluation"]["amount_tolerance"])

    model_names = [args.model] if args.model else list(cfg["models"].keys())
    metrics: dict[str, dict] = {}
    for name in model_names:
        pred_path = results_dir / f"{name}_pred.jsonl"
        if not pred_path.exists():
            print(f"跳过 {name}：无预测文件 {pred_path}")
            continue
        m = evaluate_model(pred_path, tol)
        metrics[name] = m
        out = results_dir / f"metrics_{name}.json"
        out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] {json.dumps(m, ensure_ascii=False)}")

    if metrics:
        cmp_path = results_dir / "comparison.md"
        write_comparison(metrics, cmp_path)
        print(f"对比报告 -> {cmp_path}")


if __name__ == "__main__":
    main()
