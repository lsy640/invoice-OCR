"""FastAPI 后端：发票实付金额 / 停车信息 两个识别功能 + 静态前端托管。

启动后惰性加载一次 GLM-OCR（按 inference.backend 自动选 transformers/MLX），
InvoicePipeline 与 ParkingPipeline 共用同一 backend；推理用全局锁串行（单模型）。

运行：python app/backend/server.py   或   uvicorn server:app（在 app/backend 目录内）
"""
from __future__ import annotations

import io
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


# ── 业务执行（均在 _infer_lock 内调用）─────────────────────────────
def _run_invoice(mode, files, urls, excel_bytes) -> list[dict]:
    pipe = _state["invoice"]
    if mode == "images":
        imgs = _load_uploads(files); _check_batch(len(imgs))
        return [{**r, "source": f"上传图片#{i+1}"} for i, r in enumerate(pipe.process(imgs))]
    if mode == "urls":
        us = _split_urls(urls); _check_batch(len(us))
        return pipe.process(us)
    if mode == "excel":
        rows = excel_io.read_invoice_excel(excel_bytes); _check_batch(len(rows))
        return pipe.process([r["pic_url"] for r in rows], [r["supply_money"] for r in rows])
    raise HTTPException(400, f"未知 mode: {mode}")


def _run_parking(mode, files, urls, excel_bytes) -> list[dict]:
    pipe = _state["parking"]
    if mode == "images":
        imgs = _load_uploads(files); _check_batch(len(imgs))
        return [_flatten_parking(pipe.process(imgs), "上传图片(一组)")]
    if mode == "urls":
        us = _split_urls(urls); _check_batch(len(us))
        return [_flatten_parking(pipe.process(us), "URL(一组)")]
    if mode == "excel":
        rows = excel_io.read_parking_excel(excel_bytes); _check_batch(len(rows))
        out = []
        for i, r in enumerate(rows):
            res = pipe.process(r["images"], cost=r["cost"], vin=r["vin"])
            out.append(_flatten_parking(res, r.get("workflow_no") or f"第{i+1}行"))
        return out
    raise HTTPException(400, f"未知 mode: {mode}")


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
    try:
        with _infer_lock:
            if task == "invoice":
                results, columns = _run_invoice(mode, files, urls, excel_bytes), INVOICE_COLUMNS
            else:
                results, columns = _run_parking(mode, files, urls, excel_bytes), PARKING_COLUMNS
    except ValueError as e:
        raise HTTPException(400, str(e))
    for r in results:                      # raw/refine_raw 仅调试用，不回传前端
        r.pop("raw", None)
        r.pop("refine_raw", None)
    n_mismatch = sum(1 for r in results if r.get("amount_match") == "✗" or r.get("vin_match") == "✗")
    return {"task": task, "mode": mode, "columns": columns,
            "results": results, "n": len(results), "n_mismatch": n_mismatch}


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
