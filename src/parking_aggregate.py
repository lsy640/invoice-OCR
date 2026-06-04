"""按行聚合逐图抽取结果，计算停车时长与单价，合并原始表导出 Excel。

聚合规则（每个 workflow_no）：
- 进出场时间：取所有「水印照片」识别到的有效时间，最小值=进场、最大值=出场；
  停车时长(小时) = 出场 - 进场；停车单价(元/小时) = cost / 时长。
- 车架号：水印照片中出现最多的有效 17 位车架号。
- 缴费信息：支付凭证识别到的金额（多张取合计）、支付时间、车牌号。
- 车牌号：所有图片中识别到的有效车牌（优先支付凭证，其次水印）。

输出两个 sheet：
- 汇总：原始表全部列 + 识别与计算结果 + 与标签(vin/cost)的匹配标记。
- 逐图明细：每张图的类型与识别字段。

用法： python src/parking_aggregate.py
"""
from __future__ import annotations

import collections
import json

import pandas as pd

from common import load_config, resolve


def _mode(values: list):
    values = [v for v in values if v]
    if not values:
        return None
    return collections.Counter(values).most_common(1)[0][0]


def aggregate_group(imgs: list[dict], cost: float | None = None,
                    gt_vin: str | None = None, tol: float = 0.01) -> dict:
    """把一组图片的逐图记录聚合为一行结果（进出场时间/时长/单价/车架号/金额/车牌等）。

    单一真源：批处理(main)与单组 pipeline 共用。imgs 为 record_from_kie 产出的逐图字典列表。
    """
    wms = [r for r in imgs if r.get("img_type") == "watermark"]
    pays = [r for r in imgs if r.get("img_type") == "payment"]
    invoices = [r for r in imgs if r.get("img_type") == "invoice"]
    inv_with_lease = [r for r in invoices if r.get("lease_start") and r.get("lease_end")]

    # 进出场时间：有电子发票且备注解析出停车区间则以发票为准，否则用非支付/非发票照片的最早/最晚时间
    if inv_with_lease:
        inv = inv_with_lease[0]
        entry = pd.to_datetime(inv["lease_start"])
        exit_ = pd.to_datetime(inv["lease_end"])
        time_source = "invoice"
    else:
        park_photos = [r for r in imgs if r.get("img_type") not in ("payment", "invoice") and r.get("dt")]
        t = sorted([pd.to_datetime(r["dt"]) for r in park_photos])
        entry = t[0] if t else None
        exit_ = t[-1] if t else None
        time_source = "watermark" if t else ""
    hours = round((exit_ - entry).total_seconds() / 3600, 3) if (entry is not None and exit_ is not None and exit_ > entry) else None
    unit_price = round(cost / hours, 2) if (cost is not None and hours and hours > 0) else None

    vin_det = _mode([r.get("vin") for r in (wms + invoices)]) or _mode([r.get("vin") for r in imgs])
    gt_vin = gt_vin.upper() if gt_vin else None
    vin_match = "✓" if (vin_det and gt_vin and vin_det.upper() == gt_vin) else ("✗" if (imgs and gt_vin) else "")

    amt_srcs = invoices or pays
    amts = [r["amount"] for r in amt_srcs if r.get("amount") is not None]
    pay_amount = round(max(amts), 2) if amts else None
    amount_match = ""
    if cost is not None and pay_amount is not None:
        amount_match = "✓" if abs(pay_amount - cost) <= tol else "✗"
    pay_time = _mode([r.get("dt") for r in amt_srcs]) or (amt_srcs[0].get("dt") if amt_srcs else None)
    plate = (_mode([r.get("plate") for r in invoices]) or _mode([r.get("plate") for r in pays])
             or _mode([r.get("plate") for r in imgs]))

    return {
        "num_images": len(imgs), "num_watermark": len(wms), "num_payment": len(pays),
        "num_invoice": len(invoices), "time_source": time_source,
        "entry_time": str(entry) if entry is not None else "",
        "exit_time": str(exit_) if exit_ is not None else "",
        "park_hours": hours, "unit_price_yuan_per_h": unit_price,
        "vin_detected": vin_det or "", "vin_match": vin_match,
        "payment_amount": pay_amount, "amount_match": amount_match,
        "payment_time": pay_time or "", "plate": plate or "",
    }


def main() -> None:
    cfg = load_config()
    pk = cfg["parking"]
    pred_path = resolve(pk["pred_jsonl"])
    if not pred_path.exists():
        raise SystemExit(f"未找到逐图抽取结果 {pred_path}")

    # 逐图结果按 workflow_no 分组
    by_wf: dict[str, list[dict]] = collections.defaultdict(list)
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_wf[str(r["workflow_no"])].append(r)

    df = pd.read_excel(resolve(pk["xlsx"]))
    tol = float(cfg["evaluation"]["amount_tolerance"])

    agg_cols = {c: [] for c in [
        "num_images", "num_watermark", "num_payment", "num_invoice",
        "time_source", "entry_time", "exit_time", "park_hours", "unit_price_yuan_per_h",
        "vin_detected", "vin_match", "payment_amount", "amount_match",
        "payment_time", "plate",
    ]}
    detail_rows = []

    for _, row in df.iterrows():
        wf = str(row["workflow_no"])
        imgs = by_wf.get(wf, [])
        cost = None if pd.isna(row.get("cost")) else float(row["cost"])
        gt_vin = None if pd.isna(row.get("vin")) else str(row["vin"])
        res = aggregate_group(imgs, cost=cost, gt_vin=gt_vin, tol=tol)
        for c in agg_cols:
            agg_cols[c].append(res[c])

        for r in imgs:
            detail_rows.append({
                "workflow_no": wf, "image_id": r.get("image_id"),
                "img_type": r.get("img_type"), "dt": r.get("dt"),
                "vin": r.get("vin"), "plate": r.get("plate"), "amount": r.get("amount"),
                "lease_start": r.get("lease_start"), "lease_end": r.get("lease_end"),
                "error": r.get("error"), "raw": r.get("raw"),
            })

    for c, vals in agg_cols.items():
        df[c] = vals

    out = resolve(cfg["paths"]["results_dir"]) / "停车识别结果.xlsx"
    with pd.ExcelWriter(out) as writer:
        df.to_excel(writer, sheet_name="汇总", index=False)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="逐图明细", index=False)

    n = len(df)
    vin_ok = sum(1 for x in agg_cols["vin_match"] if x == "✓")
    amt_ok = sum(1 for x in agg_cols["amount_match"] if x == "✓")
    has_price = sum(1 for x in agg_cols["unit_price_yuan_per_h"] if x is not None)
    print(f"已导出 {out}")
    print(f"  共 {n} 行 | 车架号匹配 {vin_ok} | 缴费金额匹配 {amt_ok} | 可算出单价 {has_price}")


if __name__ == "__main__":
    main()
