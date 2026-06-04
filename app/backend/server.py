"""FastAPI 后端：发票实付金额 / 停车信息 两个识别功能 + 静态前端托管。

启动后惰性加载一次 GLM-OCR（按 inference.backend 自动选 transformers/MLX），
InvoicePipeline 与 ParkingPipeline 共用同一 backend；推理用全局锁串行（单模型）。

运行：python app/backend/server.py   或   uvicorn server:app（在 app/backend 目录内）
"""
from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# 让 src/ 与本目录可导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../OCR
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND = PROJECT_ROOT / "app" / "frontend"
for p in (str(PROJECT_ROOT / "src"), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import excel_io                                            # noqa: E402
from common import load_config                             # noqa: E402
from schemas import INVOICE_COLUMNS, PARKING_COLUMNS       # noqa: E402

_CFG = load_config()
_APP = _CFG.get("app", {})
_MAX_BATCH = int(_APP.get("max_batch_images", 500))

app = FastAPI(title="GLM-OCR 票据/停车识别")

# ── 惰性加载推理后端与两个管线（共用一个 backend）+ 推理串行锁 ────────
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_state: dict = {"backend": None, "invoice": None, "parking": None, "name": None}


def _ensure_loaded():
    if _state["invoice"] is not None:
        return
    with _load_lock:
        if _state["invoice"] is not None:
            return
        from inference import get_backend
        from invoice_pipeline import InvoicePipeline
        from parking_pipeline import ParkingPipeline
        backend = get_backend(_CFG)
        _state.update(backend=backend, name=backend.name,
                      invoice=InvoicePipeline(backend=backend),
                      parking=ParkingPipeline(backend=backend))


# ── 输入解析辅助 ───────────────────────────────────────────────────
def _split_urls(text: str) -> list[str]:
    out = []
    for line in (text or "").replace(",", "\n").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def _load_uploads(files: list[UploadFile]) -> list[Image.Image]:
    return [Image.open(io.BytesIO(f.file.read())).convert("RGB") for f in files]


def _check_batch(n: int):
    if n == 0:
        raise HTTPException(400, "未提供任何图片/URL/数据")
    if n > _MAX_BATCH:
        raise HTTPException(400, f"数量 {n} 超过上限 {_MAX_BATCH}")


def _flatten_parking(res: dict, source: str) -> dict:
    out = {k: v for k, v in res.items() if k != "images"}
    out["source"] = source
    imgs = res.get("images", [])
    errs = [im for im in imgs if im.get("error")]
    out["error"] = "" if not errs else f"{len(errs)}/{len(imgs)} 图异常"
    return out


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 路由 ───────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _state["invoice"] is not None, "backend": _state["name"]}


@app.post("/api/recognize")
def recognize(
    task: str = Form(...),
    mode: str = Form(...),
    urls: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    excel: UploadFile | None = File(default=None),
):
    if task not in ("invoice", "parking"):
        raise HTTPException(400, "task 须为 invoice 或 parking")
    if mode not in ("images", "urls", "excel"):
        raise HTTPException(400, "mode 须为 images/urls/excel")

    _ensure_loaded()
    excel_bytes = excel.file.read() if excel is not None else b""

    def generate_stream():
        with _infer_lock:
            yield _sse_event("status", {"phase": "processing"})

            try:
                if task == "invoice":
                    columns = INVOICE_COLUMNS
                    pipe = _state["invoice"]
                    if mode == "images":
                        imgs = _load_uploads(files); _check_batch(len(imgs))
                        sources = [f"上传图片#{i+1}" for i in range(len(imgs))]
                        labels = [None] * len(imgs)
                    elif mode == "urls":
                        us = _split_urls(urls); _check_batch(len(us))
                        imgs, sources = us, list(us)
                        labels = [None] * len(us)
                    elif mode == "excel":
                        rows = excel_io.read_invoice_excel(excel_bytes); _check_batch(len(rows))
                        imgs = [r["pic_url"] for r in rows]
                        sources, labels = list(imgs), [r["supply_money"] for r in rows]
                    else:
                        yield _sse_event("error", {"detail": f"未知 mode: {mode}"})
                        return

                    total = len(imgs)
                    results = []
                    for i, (src, lb) in enumerate(zip(imgs, labels)):
                        yield _sse_event("progress", {"current": i, "total": total})
                        r = pipe.process_one(src, lb)
                        if mode == "images" and isinstance(src, Image.Image):
                            r["source"] = sources[i]
                        results.append(r)
                    yield _sse_event("progress", {"current": total, "total": total})

                else:
                    columns = PARKING_COLUMNS
                    pipe = _state["parking"]
                    if mode == "images":
                        imgs = _load_uploads(files); _check_batch(len(imgs))
                        total_imgs = len(imgs)
                        yield _sse_event("progress", {"current": 0, "total": total_imgs, "stage": "逐图识别"})
                        # Process images with per-image progress
                        recs, pil_imgs = [], []
                        for idx, src in enumerate(imgs):
                            try:
                                im = pipe.load_image(src)
                                pil_imgs.append(im)
                                r = pipe.extract_image(im)
                            except Exception as e:
                                im = None
                                r = {"img_type": None, "error": f"load_error:{type(e).__name__}:{e}",
                                     "dt": None, "vin": None, "plate": None, "amount": None,
                                     "lease_start": None, "lease_end": None, "raw": None}
                            r["image"] = str(src) if not isinstance(src, Image.Image) else f"上传图片#{idx+1}"
                            recs.append(r)
                            yield _sse_event("progress", {"current": idx + 1, "total": total_imgs, "stage": "逐图识别"})
                        # refine
                        for im, r in zip(pil_imgs, recs):
                            if im is not None and r.get("img_type") == "invoice" and not (r.get("lease_start") and r.get("lease_end")):
                                pipe.refine_invoice(im, r)
                        from parking_aggregate import aggregate_group
                        summary = aggregate_group(recs, cost=None, gt_vin=None, tol=pipe.tol)
                        res = {"cost": None, "vin_label": None, **summary, "images": recs}
                        results = [_flatten_parking(res, "上传图片(一组)")]

                    elif mode == "urls":
                        us = _split_urls(urls); _check_batch(len(us))
                        total_imgs = len(us)
                        recs, pil_imgs = [], []
                        for idx, src in enumerate(us):
                            yield _sse_event("progress", {"current": idx, "total": total_imgs, "stage": "逐图识别"})
                            try:
                                im = pipe.load_image(src)
                                pil_imgs.append(im)
                                r = pipe.extract_image(im)
                            except Exception as e:
                                im = None
                                r = {"img_type": None, "error": f"load_error:{type(e).__name__}:{e}",
                                     "dt": None, "vin": None, "plate": None, "amount": None,
                                     "lease_start": None, "lease_end": None, "raw": None}
                            r["image"] = str(src)
                            recs.append(r)
                        yield _sse_event("progress", {"current": total_imgs, "total": total_imgs, "stage": "逐图识别"})
                        for im, r in zip(pil_imgs, recs):
                            if im is not None and r.get("img_type") == "invoice" and not (r.get("lease_start") and r.get("lease_end")):
                                pipe.refine_invoice(im, r)
                        from parking_aggregate import aggregate_group
                        summary = aggregate_group(recs, cost=None, gt_vin=None, tol=pipe.tol)
                        res = {"cost": None, "vin_label": None, **summary, "images": recs}
                        results = [_flatten_parking(res, "URL(一组)")]

                    elif mode == "excel":
                        rows = excel_io.read_parking_excel(excel_bytes); _check_batch(len(rows))
                        total = len(rows)
                        results = []
                        for i, r in enumerate(rows):
                            yield _sse_event("progress", {"current": i, "total": total, "stage": r.get("workflow_no") or f"第{i+1}行"})
                            res = pipe.process(r["images"], cost=r["cost"], vin=r["vin"])
                            results.append(_flatten_parking(res, r.get("workflow_no") or f"第{i+1}行"))
                        yield _sse_event("progress", {"current": total, "total": total})
                    else:
                        yield _sse_event("error", {"detail": f"未知 mode: {mode}"})
                        return

            except ValueError as e:
                yield _sse_event("error", {"detail": str(e)})
                return
            except HTTPException as e:
                yield _sse_event("error", {"detail": e.detail})
                return

            for r in results:
                r.pop("raw", None)
                r.pop("refine_raw", None)
            n_mismatch = sum(1 for r in results if r.get("amount_match") == "✗" or r.get("vin_match") == "✗")
            yield _sse_event("done", {"task": task, "mode": mode, "columns": columns,
                                      "results": results, "n": len(results), "n_mismatch": n_mismatch})

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/api/export")
def export(payload: dict):
    task = payload.get("task")
    results = payload.get("results") or []
    columns = payload.get("columns") or (INVOICE_COLUMNS if task == "invoice" else PARKING_COLUMNS)
    data = excel_io.results_to_xlsx(results, columns=columns)
    fname = f"{'发票' if task == 'invoice' else '停车'}识别结果.xlsx"
    # HTTP 头只能 latin-1，中文文件名需 RFC5987 百分号编码
    cd = f"attachment; filename=result.xlsx; filename*=UTF-8''{quote(fname)}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": cd},
    )


@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host=_APP.get("host", "0.0.0.0"), port=int(_APP.get("port", 8000)))


if __name__ == "__main__":
    main()
