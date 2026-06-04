"""KIE 输出解析与字段归一化（与推理后端无关，供 transformers / vllm 两条路径复用）。"""
from __future__ import annotations

import json
import re

TYPE_TO_INT = {"加油": 0, "充电": 1}


def parse_json(text: str) -> dict | None:
    """从模型输出中尽力解析出 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def norm_amount(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"[^\d.]", "", str(val))
    # 处理多个小数点等异常
    if s.count(".") > 1:
        head, _, tail = s.partition(".")
        s = head + "." + tail.replace(".", "")
    try:
        return float(s) if s and s != "." else None
    except ValueError:
        return None


def norm_type(val) -> int | None:
    """把模型给出的「补能类型」文本归一化为 0=加油 / 1=充电；无法判断返回 None。

    GLM-OCR 的 KIE 槽常被填入就近文本（充电量「37.89度」、「团油/中石油」等），
    故需用关键词与单位兜底。充电票不含「油」、加油票不含「充电/度数」，键词无冲突。
    """
    if val is None:
        return None
    s = str(val)
    # 显式类型词优先
    if "充电" in s:
        return 1
    if "加油" in s:
        return 0
    # 充电线索：度(数)/kWh/电量/充电桩站
    if re.search(r"\d\s*度|度电|kwh|千瓦时|电量|充电桩|充电站|直流|快充|超充|桩", s, re.I):
        return 1
    # 加油线索：各种「油」与升数(中/英 升、L)、油站品牌
    if re.search(r"汽油|柴油|团油|加油站|油枪|油品|中石油|中石化|用油|加油卡|\d\s*升|\d\s*[lL]\b|油", s):
        return 0
    return None


def type_from_unit(s) -> int | None:
    """仅依据物理计量单位判类型：度/kWh → 充电；升/L → 加油。比模型分类更可信。"""
    if not s:
        return None
    if re.search(r"\d\s*度|度电|kwh|千瓦时", str(s), re.I):
        return 1
    if re.search(r"\d\s*升|\d\s*[lL]\b", str(s)):
        return 0
    return None


def fields_from_parsed(d: dict) -> tuple[float | None, int | None]:
    """从解析出的 JSON dict 中按键名（中/英文均可）抽取 amount 与 type。

    若「补能类型」槽为空/无法判断，则用「充电量/加油量」槽的值（37.89度 / 35升）兜底推断类型。
    """
    amount = None
    type_slot = qty_val = None
    for k, v in d.items():
        ks = str(k)
        if amount is None and re.search(r"实付|金额|amount|付款|支付金额", ks, re.I):
            amount = norm_amount(v)
        # 数量槽（键名含 充电量/加油量/电量/油量）须先于类型槽判断（其键名也含「充电/加油」字样）
        if re.search(r"充电量|加油量|电量|油量|度.*升|升.*度", ks):
            qty_val = str(v)
        elif re.search(r"补能|类型|type", ks, re.I):
            type_slot = str(v)
    # 类型判定优先级：物理单位(度/升) > 类型槽关键词 > 数量槽其他线索
    type_ = type_from_unit(qty_val)
    if type_ is None and type_slot:
        type_ = norm_type(type_slot)
    if type_ is None and qty_val:
        type_ = norm_type(qty_val)
    return amount, type_
