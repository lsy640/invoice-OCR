"""抽取指令与 JSON Schema（仅两个字段：实付金额 + 补能类型）。"""
from __future__ import annotations

# vLLM guided decoding 用的 JSON Schema：强制模型输出结构化结果。
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {
            "type": ["number", "null"],
            "description": "实付总金额，单位元；票据上的实际支付/实收金额",
        },
        "supplementation_type": {
            "type": ["string", "null"],
            "enum": ["加油", "充电", None],
            "description": "补能类型，只能是 加油 或 充电",
        },
    },
    "required": ["amount", "supplementation_type"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = "你是票据信息抽取助手，只输出 JSON，不要解释、不要 Markdown 代码块。"

USER_PROMPT = (
    "这是一张车辆补能（加油或充电）的发票或支付截图。请抽取以下两个字段并输出 JSON：\n"
    "1. amount：实付总金额（单位：元，数字）。若票据同时出现订单金额、优惠、实付等多个金额，"
    "取用户实际支付/实收的总金额；无法判断则为 null。\n"
    "2. supplementation_type：补能类型，只能取 \"加油\" 或 \"充电\"。"
    "加油站/汽油/柴油/升数/油枪等 → 加油；充电站/充电桩/度数(kWh)/电费/服务费 → 充电；"
    "无法判断则为 null。\n"
    "严格按如下格式仅输出一个 JSON 对象：\n"
    '{"amount": 数字或null, "supplementation_type": "加油"或"充电"或null}'
)


# ── GLM-OCR 专用 KIE prompt ────────────────────────────────────────
# GLM-OCR 的信息抽取必须用「请按下列JSON格式输出图中信息: + 空值JSON模板」的固定范式
# （见模型卡 Prompt Limited 段）。键用中文以贴合其训练分布；模型会把值填进去。
GLM_KIE_PROMPT = (
    "请按下列JSON格式输出图中信息:\n"
    "{\n"
    '    "实付金额": "",\n'
    '    "充电量(度)或加油量(升)": "",\n'
    '    "补能类型(加油或充电)": ""\n'
    "}"
)
# 说明：
# - "实付金额" 取用户实际支付的总金额（含充电费+服务费的合计，非充值面额、非单项费用）。
# - 增设 "充电量(度)或加油量(升)" 独立槽：避免模型把「37.89度/35升」误填进「补能类型」，
#   同时该值(度/升)是补能类型的强信号，供解析兜底（见 kie_common.fields_from_parsed）。


# ── 任务一·发票/支付凭证：只抽实付金额（不抽补能类型）────────────────
# 价税合计是电子发票的总额信号；增设独立槽以便取含税总额（与停车任务口径一致）。
GLM_AMOUNT_PROMPT = (
    "请按下列JSON格式输出图中信息:\n"
    "{\n"
    '    "实付金额": "",\n'
    '    "价税合计": ""\n'
    "}"
)


# ── 停车数据 GLM-OCR prompt（统一覆盖两类图：进出场水印 / 支付凭证）─────
# 水印照片：左下角含 日期+时间、完整车架号(17位)、有时含车牌；
# 支付凭证：含 支付金额、转账/支付时间、有时含车牌。
# 用一个 schema 全覆盖，缺失项留空；图片类型由后处理按字段存在性判定。
GLM_PARKING_PROMPT = (
    "请按下列JSON格式输出图中信息:\n"
    "{\n"
    '    "日期": "",\n'
    '    "时间": "",\n'
    '    "完整车架号": "",\n'
    '    "车牌号": "",\n'
    '    "支付金额": "",\n'
    '    "价税合计": "",\n'
    '    "备注": ""\n'
    "}"
)
# - 日期/时间拆两槽：水印照片时间常是大字号(如 19:01)、与日期分离，单列利于读取。
# - 支付金额：支付截图的实付金额；价税合计：电子发票的价税合计(小写总额) —— 模型读得最稳，
#   是判定「电子发票」的可靠信号。
# - 备注：电子发票「备注」框内的全部文字（含 车架号、车牌、以及停车时间/租赁期的日期区间，
#   不同发票标签各异：「停车时间（…-…）」「租赁期起止：…」或直接一段日期范围）。让模型整段读出，
#   再由后处理鲁棒解析日期区间(parse_lease_period)，作为停车进出场时间的权威来源。


def build_glm_text_messages(image, instruction: str = "Text Recognition:") -> list[dict]:
    """GLM-OCR 全文识别模式（文档解析），用于二次 OCR 裁剪后的发票备注区域。"""
    img_part = {"type": "image", "image": image} if not isinstance(image, str) else {"type": "image", "url": image}
    return [
        {"role": "user", "content": [img_part, {"type": "text", "text": instruction}]},
    ]


def build_glm_parking_messages(image) -> list[dict]:
    """构造停车图片的 GLM-OCR chat 消息。image 可为 PIL.Image 或本地路径字符串。"""
    img_part = {"type": "image", "image": image} if not isinstance(image, str) else {"type": "image", "url": image}
    return [
        {"role": "user", "content": [img_part, {"type": "text", "text": GLM_PARKING_PROMPT}]},
    ]


def build_glm_messages(image) -> list[dict]:
    """构造 GLM-OCR（transformers）chat 消息。image 可为 PIL.Image 或本地路径字符串。"""
    img_part = {"type": "image", "image": image} if not isinstance(image, str) else {"type": "image", "url": image}
    return [
        {"role": "user", "content": [img_part, {"type": "text", "text": GLM_KIE_PROMPT}]},
    ]


def build_messages(image_data_url: str) -> list[dict]:
    """构造 OpenAI chat.completions 的 messages（图片以 data URL 内联）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
