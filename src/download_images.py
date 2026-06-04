"""下载 records.json 中的图片到本地缓存目录，可断点续跑。

- 每张图存为 <image_dir>/<workflow_no>.<ext>
- 已存在且非空的文件直接跳过
- 下载后用 PIL 校验可解码；若长边超过 max_image_long_side 则等比缩小覆盖保存
  （缩小可降低 VLM 的图像 token 数与显存占用）

用法：
  python src/download_images.py [--records data/records.json] [--workers 8]
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

# 复用一个 session（连接池）
_session = requests.Session()


def _ext_from_url(url: str) -> str:
    tail = url.split("?")[0].rsplit(".", 1)
    ext = tail[-1].lower() if len(tail) == 2 else ""
    return ext if ext in {"png", "jpg", "jpeg", "webp", "bmp"} else "png"


def image_path(image_dir: Path, workflow_no: str, url: str) -> Path:
    return image_dir / f"{workflow_no}.{_ext_from_url(url)}"


def _download_one(rec: dict, image_dir: Path, max_long_side: int, timeout: float) -> tuple[str, str]:
    """返回 (workflow_no, status)。status ∈ {skip, ok, resized, error:...}"""
    wf = rec["workflow_no"]
    url = rec["pic_url"]
    dst = image_path(image_dir, wf, url)
    if dst.exists() and dst.stat().st_size > 0:
        return wf, "skip"
    try:
        resp = _session.get(url, timeout=timeout)
        resp.raise_for_status()
        raw = resp.content
        img = Image.open(io.BytesIO(raw))
        img.load()
        long_side = max(img.size)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
            img = img.convert("RGB").resize(new_size, Image.LANCZOS)
            img.save(dst)
            return wf, "resized"
        dst.write_bytes(raw)
        return wf, "ok"
    except Exception as e:  # noqa: BLE001 — 单条失败不应中断整体
        return wf, f"error:{type(e).__name__}:{e}"


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(resolve(cfg["data"]["records_json"])))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    image_dir = resolve(cfg["data"]["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)
    max_long_side = int(cfg["data"]["max_image_long_side"])

    with open(args.records, "r", encoding="utf-8") as f:
        records = json.load(f)

    counts: dict[str, int] = {"skip": 0, "ok": 0, "resized": 0, "error": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_download_one, r, image_dir, max_long_side, 30.0) for r in records]
        for i, fut in enumerate(as_completed(futs), 1):
            wf, status = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            if key == "error":
                errors.append(f"{wf} {status}")
            if i % 100 == 0:
                print(f"  进度 {i}/{len(records)}  {counts}")

    print(f"完成：{counts}  共 {len(records)} 条 -> {image_dir}")
    if errors:
        print(f"失败 {len(errors)} 条，示例：")
        for line in errors[:10]:
            print("   ", line)


if __name__ == "__main__":
    main()
