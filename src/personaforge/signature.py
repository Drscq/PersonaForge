from __future__ import annotations

import re

from personaforge.schema import Signature, UserLog

EN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def extract_signature(log: UserLog) -> Signature:
    text = log.user
    language = log.language
    return Signature(
        language=language,
        intent=classify_intent(text),
        length_bucket=length_bucket(text, language),
        politeness=classify_politeness(text),
        topic=classify_topic(text),
    )


def classify_intent(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["function", "python", "code", "函数", "代码"]):
        return "coding"
    if any(word in lowered for word in ["explain", "derivative", "probability", "解释", "概率", "导数"]):
        return "learning"
    if any(word in lowered for word in ["plan", "interview", "anxious", "申请", "计划", "焦虑", "实习"]):
        return "advice"
    if any(word in lowered for word in ["write", "message", "email", "birthday", "写", "邮件", "感谢"]):
        return "writing"
    return "general"


def classify_topic(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["python", "function", "函数", "字符串"]):
        return "programming"
    if any(word in lowered for word in ["derivative", "probability", "导数", "概率"]):
        return "math"
    if any(word in lowered for word in ["interview", "internship", "实习", "申请"]):
        return "career"
    if any(word in lowered for word in ["birthday", "email", "邮件", "感谢"]):
        return "communication"
    return "general"


def classify_politeness(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["please", "could you", "would you", "请", "帮我", "感谢"]):
        return "polite"
    return "direct"


def length_bucket(text: str, language: str) -> str:
    length = len(_tokens(text, language))
    if length <= 10:
        return "short"
    if length <= 24:
        return "medium"
    return "long"


def _tokens(text: str, language: str) -> list[str]:
    if language == "zh":
        cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
        latin = EN_TOKEN_RE.findall(text)
        return cjk + latin
    return EN_TOKEN_RE.findall(text)

