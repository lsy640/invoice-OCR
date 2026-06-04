"""停车图片字段解析：日期时间归一化、车架号/车牌校验、图片类型判定。"""
from __future__ import annotations

import re

import pandas as pd

# 17 位车架号：前后非字母数字边界，且必须含字母（排除 32 位纯数字转账单号被误截 17 位）
_VIN_RE = re.compile(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])")
# 中国车牌（含临牌/使领等后缀），如 渝A0B52试 / 川A12345
_PROVINCE = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
_PLATE_RE = re.compile(rf"[{_PROVINCE}][A-Z][A-Z0-9]{{4,6}}[试学警港澳挂]?")


def parse_datetime(s):
    """从文本中提取日期(YYYY-MM-DD)与时间(HH:MM[:SS])并组合为 Timestamp；失败返回 None。

    用正则分别抠日期/时间，可容忍 星期/天气/温度等噪声。
    支持 2025-12-31 19:01 / 2025.12.31 星期三 / 2026年1月4日 10:54:31 / 2026-01-04 等。
    """
    if s is None:
        return None
    t = str(s)
    t = t.replace("年", "-").replace("月", "-").replace("日", " ").replace("/", "-").replace(".", "-")
    dm = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if not dm:
        return None
    tm = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", t)
    y, mo, d = (int(x) for x in dm.groups())
    if tm:
        hh, mi = int(tm.group(1)), int(tm.group(2))
        ss = int(tm.group(3)) if tm.group(3) else 0
    else:
        hh = mi = ss = 0
    try:
        return pd.Timestamp(y, mo, d, hh, mi, ss)
    except Exception:  # noqa: BLE001
        return None


_LEASE_DT_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*(?:(\d{1,2}):(\d{2}))?")


def parse_lease_period(s):
    """从发票备注/日期文本中提取停车起止日期区间，返回 (start, end)。

    兼容多种写法：
      '停车时间（2026年1月20日-2026年2月19日）'、'2026年2月28日-2026年3月27日'(裸范围)、
      '2026.04.15-05.14'(结束日期缺年份，继承起始年份)、'2026-03-01 10:15 2026-03-10 10:15'。
    需 >=2 个日期(或 1 个完整日期 + 1 个缺年份结束日期)才返回区间，否则 (单日期, None) 或 (None, None)。
    """
    if s is None:
        return None, None
    t = str(s).replace("年", "-").replace("月", "-").replace("日", " ")
    t = t.replace("/", "-").replace(".", "-").replace("．", "-")
    t = re.sub(r"\s+", " ", t)
    dts = []
    full = list(re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})\s*(?:(\d{1,2}):(\d{2}))?", t))
    for m in full:
        y, mo, d, hh, mi = m.groups()
        try:
            dts.append(pd.Timestamp(int(y), int(mo), int(d), int(hh or 0), int(mi or 0)))
        except Exception:  # noqa: BLE001
            continue
    # 结束日期缺年份：最后一个完整日期之后形如 "-MM-DD"，继承其年份
    if full:
        last = full[-1]
        m2 = re.match(r"\s*[-~至到]+\s*(\d{1,2})-(\d{1,2})(?!\s*\d)", t[last.end():])
        if m2:
            try:
                dts.append(pd.Timestamp(int(last.group(1)), int(m2.group(1)), int(m2.group(2))))
            except Exception:  # noqa: BLE001
                pass
    dts = sorted(set(dts))
    if len(dts) >= 2:
        return dts[0], dts[-1]
    return (dts[0], None) if dts else (None, None)


def extract_vin(s) -> str | None:
    """提取 17 位车架号；要求含字母，避免把纯数字单号当成 VIN。"""
    if s is None:
        return None
    up = str(s).upper().replace(" ", "").replace("·", "")
    for m in _VIN_RE.finditer(up):
        c = m.group(0)
        if sum(ch.isalpha() for ch in c) >= 2:  # 真实 VIN 含多个字母
            return c
    return None


def extract_plate(s) -> str | None:
    if s is None:
        return None
    t = re.sub(r"[ ·•\-]", "", str(s))
    m = _PLATE_RE.search(t)
    return m.group(0) if m else None


def to_amount(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = re.sub(r"[^\d.]", "", str(s))
    if t.count(".") > 1:
        head, _, tail = t.partition(".")
        t = head + "." + tail.replace(".", "")
    try:
        return float(t) if t and t != "." else None
    except ValueError:
        return None


def classify(vin: str | None, amount: float | None, dt) -> str:
    """判定图片类型：watermark(进出场水印) / payment(支付凭证) / unknown。"""
    if vin:
        return "watermark"
    if amount is not None:
        return "payment"
    return "unknown"


def record_from_kie(parsed: dict) -> dict:
    """把 GLM-OCR KIE 的 JSON 输出统一解析为一张图的字段记录。

    返回 {img_type, dt, vin, plate, amount, lease_start, lease_end}（值为 str/None）。
    供逐图抽取(parking_extract_hf)与单组 pipeline 复用，保证口径一致。
    """
    def g(*keys):
        for k, v in parsed.items():
            if any(t in str(k) for t in keys):
                return v
        return None

    date_s = str(g("日期") or "")
    time_s = str(g("时间") or "")
    dt = parse_datetime(f"{date_s} {time_s}".strip())
    # 金额：发票总额(价税合计,含税) 恒 >= 税前/支付金额，取两者最大值为实付总额（纠正偶发填反）
    inv_total = to_amount(g("价税合计", "税合计"))
    pay_amt = to_amount(g("支付金额", "支付"))
    amts = [x for x in (inv_total, pay_amt) if x is not None]
    amount = max(amts) if amts else None
    # 备注框文字：含 车架号/车牌/停车日期区间；主栏槽优先，缺则从备注抠
    bz = str(g("备注", "发票备注") or "")
    vin = extract_vin(g("车架")) or extract_vin(bz)
    plate = extract_plate(g("车牌")) or extract_plate(bz)
    # 停车起止：从备注解析日期区间；若备注无，看「日期」槽是否被填成区间
    ls, le = parse_lease_period(bz)
    if le is None:
        cs, ce = parse_lease_period(date_s)
        if ce is not None:
            ls, le = cs, ce
    # 判定：读到价税合计即电子发票；否则按车架号/金额常规分类
    img_type = "invoice" if (inv_total is not None or (ls is not None and le is not None)) else classify(vin, amount, dt)
    return {
        "img_type": img_type,
        "dt": None if dt is None else str(dt),
        "vin": vin,
        "plate": plate,
        "amount": amount,
        "lease_start": None if ls is None else str(ls),
        "lease_end": None if le is None else str(le),
    }
