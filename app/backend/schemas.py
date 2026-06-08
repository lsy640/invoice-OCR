"""API 数据模型：结果列定义。"""
from __future__ import annotations

# 两个任务的结果列顺序（前端表格与导出 Excel 共用）
INVOICE_COLUMNS = ["source", "pred_amount", "label", "amount_match", "error"]
PARKING_COLUMNS = [
    "source", "cost", "vin_label", "num_images", "time_source",
    "entry_time", "exit_time", "park_hours", "unit_price_yuan_per_h",
    "vin_detected", "vin_match", "payment_amount", "amount_match", "plate", "error",
]
