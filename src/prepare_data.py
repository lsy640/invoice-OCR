"""读取 发票数据.xlsx，规范化为评测记录列表并落盘为 JSON。

输出每条记录字段：
  workflow_no : str   唯一键
  vin         : str
  pic_url     : str   待识别图片链接
  gt_amount   : float 实付金额标签（supply_money）
  gt_type     : int   补能类型标签（0=加油, 1=充电）

用法：
  python src/prepare_data.py [--limit N]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import load_config, resolve


def build_records(xlsx_path, limit: int | None = None) -> list[dict]:
    df = pd.read_excel(xlsx_path)
    required = {"workflow_no", "pic_url", "supply_money", "supplementation_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"xlsx 缺少必要列: {missing}")

    records: list[dict] = []
    for _, row in df.iterrows():
        url = row["pic_url"]
        if pd.isna(url) or not str(url).strip():
            continue  # 无图片链接的行跳过
        try:
            gt_type = int(row["supplementation_type"])
        except (ValueError, TypeError):
            continue
        records.append(
            {
                "workflow_no": str(row["workflow_no"]),
                "vin": None if pd.isna(row.get("vin")) else str(row.get("vin")),
                "pic_url": str(url).strip(),
                "gt_amount": None if pd.isna(row["supply_money"]) else float(row["supply_money"]),
                "gt_type": gt_type,
            }
        )
    if limit is not None:
        records = records[:limit]
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="仅取前 N 条")
    args = ap.parse_args()

    cfg = load_config()
    xlsx_path = resolve(cfg["data"]["xlsx"])
    out_path = resolve(cfg["data"]["records_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_records(xlsx_path, args.limit)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_oil = sum(1 for r in records if r["gt_type"] == 0)
    n_charge = sum(1 for r in records if r["gt_type"] == 1)
    print(f"写入 {len(records)} 条记录 -> {out_path}")
    print(f"  加油(0)={n_oil}  充电(1)={n_charge}")


if __name__ == "__main__":
    main()
