"""下载停车图片到本地缓存（OBS 直链，按 image_id 命名），可断点续跑。

用法： python src/parking_download.py [--workers 16]
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from common import load_config, resolve

_session = requests.Session()


def image_path(image_dir: Path, image_id: str) -> Path:
    return image_dir / image_id


def _download_one(img: dict, image_dir: Path, max_long_side: int, timeout: float) -> tuple[str, str]:
    oid = img["image_id"]
    dst = image_path(image_dir, oid)
    if dst.exists() and dst.stat().st_size > 0:
        return oid, "skip"
    try:
        resp = _session.get(img["obs_url"], timeout=timeout)
        resp.raise_for_status()
        raw = resp.content
        im = Image.open(io.BytesIO(raw)); im.load()
        if max(im.size) > max_long_side:
            scale = max_long_side / max(im.size)
            im = im.convert("RGB").resize((round(im.size[0]*scale), round(im.size[1]*scale)), Image.LANCZOS)
            im.save(dst)
            return oid, "resized"
        dst.write_bytes(raw)
        return oid, "ok"
    except Exception as e:  # noqa: BLE001
        return oid, f"error:{type(e).__name__}"


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pk = cfg["parking"]
    image_dir = resolve(pk["image_dir"]); image_dir.mkdir(parents=True, exist_ok=True)
    max_long_side = int(cfg["data"]["max_image_long_side"])

    with open(resolve(pk["records_json"]), "r", encoding="utf-8") as f:
        records = json.load(f)
    # 去重所有图片
    imgs = {}
    for r in records:
        for im in r["images"]:
            imgs[im["image_id"]] = im
    imgs = list(imgs.values())

    counts = {"skip": 0, "ok": 0, "resized": 0, "error": 0}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_download_one, im, image_dir, max_long_side, 30.0) for im in imgs]
        for i, fut in enumerate(as_completed(futs), 1):
            oid, status = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            if key == "error":
                errors.append(f"{oid} {status}")
            if i % 200 == 0:
                print(f"  进度 {i}/{len(imgs)} {counts}")
    print(f"完成：{counts} 共 {len(imgs)} 张 -> {image_dir}")
    if errors:
        print(f"失败 {len(errors)} 张，示例：", errors[:5])


if __name__ == "__main__":
    main()
