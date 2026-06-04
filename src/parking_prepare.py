"""读取 停车数据.xlsx，规范化为评测记录（每行含多张图），落盘 JSON。

cost_images 为逗号分隔的 ptmapi 代理 URL；对象同时存在于公有 OBS 桶，
取出对象 id 后改用 OBS 直链下载（无需 token）。

输出每条记录：
  workflow_no, vin(车架号), cost(停车费用), createtime,
  images: [ {image_id, obs_url} ... ]   # 该行的全部图片

用法： python src/parking_prepare.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from common import load_config, resolve

_ID_RE = re.compile(r"/([0-9a-fA-F]{16,}\.(?:jpg|jpeg|png))", re.I)


def extract_image_id(url: str) -> str | None:
    m = _ID_RE.search(url)
    return m.group(1) if m else None


def build_records(xlsx_path, obs_base: str, limit: int | None) -> list[dict]:
    df = pd.read_excel(xlsx_path)
    records: list[dict] = []
    for _, row in df.iterrows():
        imgs_raw = row.get("cost_images")
        if pd.isna(imgs_raw) or not str(imgs_raw).strip():
            continue
        images = []
        for u in str(imgs_raw).split(","):
            u = u.strip()
            if not u:
                continue
            oid = extract_image_id(u)
            if oid:
                images.append({"image_id": oid, "obs_url": obs_base + oid})
        if not images:
            continue
        ct = row.get("createtime")
        records.append(
            {
                "workflow_no": str(row["workflow_no"]),
                "vin": None if pd.isna(row.get("vin")) else str(row.get("vin")),
                "cost": None if pd.isna(row.get("cost")) else float(row.get("cost")),
                "createtime": None if pd.isna(ct) else str(ct),
                "images": images,
            }
        )
    if limit is not None:
        records = records[:limit]
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    pk = cfg["parking"]
    records = build_records(resolve(pk["xlsx"]), pk["obs_base"], args.limit)
    out = resolve(pk["records_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_img = sum(len(r["images"]) for r in records)
    print(f"写入 {len(records)} 行 / 共 {n_img} 张图 -> {out}")


if __name__ == "__main__":
    main()
