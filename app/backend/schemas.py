"""API 数据模型。recognize 走 multipart 表单；export 走 JSON。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RecognizeResponse(BaseModel):
    task: str
    mode: str
    columns: list[str]
    results: list[dict[str, Any]]
    n: int
    n_mismatch: int = 0          # amount/vin 不匹配的条数（有标签时）


class ExportRequest(BaseModel):
    task: str
    results: list[dict[str, Any]]
    columns: list[str] | None = None


# 两个任务的结果列顺序（前端表格与导出 Excel 共用）
INVOICE_COLUMNS = ["source", "pred_amount", "label", "amount_match", "error"]
PARKING_COLUMNS = [
    "source", "cost", "vin_label", "num_images", "time_source",
    "entry_time", "exit_time", "park_hours", "unit_price_yuan_per_h",
    "vin_detected", "vin_match", "payment_amount", "amount_match", "plate", "error",
]
