"""把抽取结果与原始数据(发票数据.xlsx)按 workflow_no 合并，标注是否匹配，导出 Excel。

输出列（在原始表全部列之后追加）：
  pred_amount        模型识别的实付金额
  pred_type          模型识别的补能类型(0=加油,1=充电)
  pred_type_text     补能类型中文(加油/充电/未知)
  amount_match       金额是否匹配(✓/✗)，|pred-标签|<=容差
  type_match         补能类型是否匹配(✓/✗)
  all_match          两者是否都匹配(✓/✗)
  raw_output         模型原始 JSON 输出
  error              失败标记(缺图/解析失败等)

用法：
  python src/export_excel.py --model glm-ocr
输出：results/识别结果对比_<model>.xlsx
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import load_config, resolve

TYPE_TEXT = {0: "加油", 1: "充电"}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-ocr", choices=list(cfg["models"].keys()))
    args = ap.parse_args()

    tol = float(cfg["evaluation"]["amount_tolerance"])
    results_dir = resolve(cfg["paths"]["results_dir"])
    pred_path = results_dir / f"{args.model}_pred.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"未找到预测文件 {pred_path}")

    # 预测按 workflow_no 建索引
    preds: dict[str, dict] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                preds[str(r["workflow_no"])] = r

    # 原始表（保留全部原列）
    df = pd.read_excel(resolve(cfg["data"]["xlsx"]))
    if "Unnamed: 6" in df.columns:
        df = df.rename(columns={"Unnamed: 6": "备注"})

    def tick(b: bool | None) -> str:
        return "✓" if b else "✗"

    pred_amount, pred_type, pred_type_text = [], [], []
    amount_match, type_match, all_match, raw_output, error_col = [], [], [], [], []
    for _, row in df.iterrows():
        wf = str(row["workflow_no"])
        p = preds.get(wf)
        if p is None:
            pred_amount.append(None); pred_type.append(None); pred_type_text.append("")
            amount_match.append(""); type_match.append(""); all_match.append("")
            raw_output.append(""); error_col.append("未识别")
            continue
        pa, pt = p.get("pred_amount"), p.get("pred_type")
        gt_a = None if pd.isna(row["supply_money"]) else float(row["supply_money"])
        gt_t = None if pd.isna(row["supplementation_type"]) else int(row["supplementation_type"])
        a_ok = (gt_a is not None and pa is not None and abs(pa - gt_a) <= tol)
        t_ok = (gt_t is not None and pt == gt_t)
        pred_amount.append(pa)
        pred_type.append(pt)
        pred_type_text.append(TYPE_TEXT.get(pt, "未知"))
        amount_match.append(tick(a_ok))
        type_match.append(tick(t_ok))
        all_match.append(tick(a_ok and t_ok))
        raw_output.append(p.get("raw") or "")
        error_col.append(p.get("error") or "")

    df["pred_amount"] = pred_amount
    df["pred_type"] = pred_type
    df["pred_type_text"] = pred_type_text
    df["amount_match"] = amount_match
    df["type_match"] = type_match
    df["all_match"] = all_match
    df["raw_output"] = raw_output
    df["error"] = error_col

    out_path = results_dir / f"识别结果对比_{args.model}.xlsx"
    df.to_excel(out_path, index=False)

    n = len(df)
    a_hit = sum(1 for x in amount_match if x == "✓")
    t_hit = sum(1 for x in type_match if x == "✓")
    j_hit = sum(1 for x in all_match if x == "✓")
    print(f"已导出 {out_path}")
    print(f"  共 {n} 行 | 金额匹配 {a_hit} ({a_hit/n:.1%}) | 类型匹配 {t_hit} ({t_hit/n:.1%}) | 全匹配 {j_hit} ({j_hit/n:.1%})")


if __name__ == "__main__":
    main()
