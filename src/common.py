"""共享工具：配置加载与路径解析。

项目根目录约定为 OCR/（即本文件父目录的父目录）。config.yaml 中的相对路径
一律相对于项目根目录解析为绝对路径。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """读取 config.yaml 并返回 dict。"""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str | os.PathLike) -> Path:
    """把 config 中的相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
