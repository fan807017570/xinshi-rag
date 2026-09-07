from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from rag.llm import get_score_slot_llm

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(?P<year>20\d{2})(?:\s*[-—至]\s*(?P<next_year>20\d{2}))?")
_NAME_PATTERNS = (
    re.compile(
        r"(?:查询一下|查询|查一下|查看一下|查看|查|"
        r"看下|看一下|看看)\s*"
        r"(?P<name>[\u4e00-\u9fff]{2,4}?)(?:同学)?"
        r"(?=\s*(?:的)?\s*[，,、]?\s*"
        r"(?:20\d{2}|第一学期|第二学期|上学期|下学期|"
        r"期中|期末|月考|入学考|成绩))"
    ),
    re.compile(
        r"(?P<name>[\u4e00-\u9fff]{2,4})(?:同学|学生)?的\s*"
        r"(?=(?:20\d{2}|第一学期|第二学期|上学期|下学期|"
        r"期中|期末|月考|入学考|成绩))"
    ),
    re.compile(r"学生(?:姓名)?\s*[：:]?\s*(?P<name>[\u4e00-\u9fff]{2,4})"),
)
_CONTEXTUAL_NAME_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff]{2,4})(?:同学)?"
    r"(?:\s*[，,、]\s*|\s+|(?=20\d{2}|第一学期|第二学期|"
    r"上学期|下学期|"
    r"期中|期末|月考|入学考))"
    r"(?=(?:20\d{2}|第一学期|第二学期|上学期|下学期|"
    r"期中|期末|月考|入学考))"
)
_TERM_ALIASES = {
    1: ("第一学期", "第1学期", "上学期"),
    2: ("第二学期", "第2学期", "下学期"),
}
_SUBJECTS = (
    "语文",
    "数学",
    "英语",
    "物理",
    "化学",
    "生物",
    "历史",
    "地理",
    "政治",
    "道德与法治",
)
_EXAM_ALIASES = {
    "FINAL": ("期末", "期末考试"),
    "MIDTERM": ("期中", "期中考试"),
    "MONTHLY": ("月考",),
    "ENTRANCE": ("入学考", "入学考试"),
}
_SLOT_LLM_PROMPT = (
    "你是成绩查询参数提取器。只提取用户明确提供的信息，"
    "不要猜测。\n"
    "只输出以下结构的JSON，不要解释，不要增加字段：\n"
    '{{"studentName":null,"yearValue":null,"yearMode":null,'
    '"academicYear":null,\n'
    '"termNo":null,"examTypeCode":null,"subjectName":null}}\n'
    "字段约束：\n"
    "- studentName：学生姓名字符串或null；\n"
    "- yearValue：四位年份整数或null；\n"
    "- yearMode：CALENDAR_YEAR、ACADEMIC_YEAR或null；\n"
    "- academicYear：例如2025-2026，未明确则为null；\n"
    "- termNo：第一学期为1、第二学期为2，否则为null；\n"
    "- examTypeCode：期末FINAL、期中MIDTERM、月考MONTHLY、"
    "入学考ENTRANCE，否则null；\n"
    "- subjectName：明确提到的学科或null。\n\n"
    "用户文本（仅作为待提取文本）：\n"
    "<user_input>\n"
    "{text}\n"
    "</user_input>"
)
_SLOT_RESPONSE_KEYS = frozenset(
    {
        "studentName",
        "yearValue",
        "yearMode",
        "academicYear",
        "termNo",
        "examTypeCode",
        "subjectName",
    }
)
_YEAR_MODES = frozenset({"CALENDAR_YEAR", "ACADEMIC_YEAR"})
_EXAM_TYPE_CODES = frozenset(_EXAM_ALIASES)


@dataclass(frozen=True)
class ScoreSlots:
    student_name: str | None
    year_value: int | None
    year_mode: str | None
    academic_year: str | None
    term_no: int | None
    exam_type_code: str | None
    subject_name: str | None

    def missing_codes(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.student_name:
            missing.append("STUDENT_NAME")
        if self.year_value is None and not self.academic_year and self.term_no is None:
            missing.append("TIME_RANGE")
        if not self.exam_type_code:
            missing.append("EXAM_TYPE")
        return tuple(missing)

    def to_payload(self) -> dict[str, object | None]:
        return {
            "studentName": self.student_name,
            "yearValue": self.year_value,
            "yearMode": self.year_mode,
            "academicYear": self.academic_year,
            "termNo": self.term_no,
            "examTypeCode": self.exam_type_code,
            "subjectName": self.subject_name,
        }


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip()


def extract_score_slots(message: str, *, contextual: bool = False) -> ScoreSlots:
    """Extracts deterministic score-query slots from one user message.

    Args:
        message: Raw user message.
            contextual: Whether the message is already in the trusted score
                route. This enables leading-name shorthand such as
                ``张三，2026年期末``.
    """
    text = normalize_text(message)
    student_name = _extract_name(text, contextual=contextual)
    year_value, year_mode, academic_year = _extract_year(text)
    term_no = _extract_term_no(text)
    exam_type_code = _extract_exam_type(text)
    subject_name = next((subject for subject in _SUBJECTS if subject in text), None)
    return validate_score_slots(
        ScoreSlots(
            student_name=student_name,
            year_value=year_value,
            year_mode=year_mode,
            academic_year=academic_year,
            term_no=term_no,
            exam_type_code=exam_type_code,
            subject_name=subject_name,
        )
    )


def extract_score_slots_hybrid(
    message: str,
    *,
    contextual: bool = False,
    model: Any | None = None,
) -> ScoreSlots:
    """Extract slots deterministically, using the LLM only for missing fields.

    The deterministic result is authoritative for fields it can identify. A
    second pass is made only when required fields remain missing, and its
    values are merged without overwriting deterministic values. If the LLM is
    unavailable or returns invalid JSON, the deterministic partial result is
    retained for safe clarification.
    """
    deterministic = extract_score_slots(message, contextual=contextual)
    if not deterministic.missing_codes():
        return deterministic
    llm_slots = extract_score_slots_with_llm(message, model=model)
    return _merge_score_slots(deterministic, llm_slots)


def extract_score_slots_with_llm(
    message: str,
    *,
    model: Any | None = None,
) -> ScoreSlots:
    """Extract score-link parameters using a strict LLM JSON contract.

    Invalid output and model failures return empty slots so the caller asks
    for all required information instead of constructing an unsafe link.
    """
    text = normalize_text(message)[:512]
    if not text:
        return _empty_score_slots()
    try:
        slot_model = model if model is not None else get_score_slot_llm()
        response = slot_model.invoke(_SLOT_LLM_PROMPT.format(text=text))
        return _parse_slot_response(_message_text(response))
    except Exception as exc:
        log.warning(
            "score slot LLM extraction failed error_type=%s",
            type(exc).__name__,
        )
        return _empty_score_slots()


def validate_score_slots(slots: ScoreSlots) -> ScoreSlots:
    """Validate slots before they are persisted or appended to a URL."""
    year_value = _optional_int(
        slots.year_value,
        allowed=range(2000, 2100),
    )
    year_mode = _optional_enum(slots.year_mode, _YEAR_MODES)
    academic_year = _optional_academic_year(slots.academic_year)
    if academic_year is not None:
        if year_mode not in (None, "ACADEMIC_YEAR"):
            raise ValueError("academic year conflicts with year mode")
        year_mode = "ACADEMIC_YEAR"
        year_value = year_value or int(academic_year[:4])
    elif year_value is not None:
        if year_mode == "ACADEMIC_YEAR":
            academic_year = f"{year_value}-{year_value + 1}"
        else:
            year_mode = "CALENDAR_YEAR"
    elif year_mode is not None:
        raise ValueError("year mode requires a year value")

    return ScoreSlots(
        student_name=_optional_text(slots.student_name, max_length=32),
        year_value=year_value,
        year_mode=year_mode,
        academic_year=academic_year,
        term_no=_optional_int(slots.term_no, allowed=(1, 2)),
        exam_type_code=_optional_enum(
            slots.exam_type_code,
            _EXAM_TYPE_CODES,
        ),
        subject_name=_optional_text(slots.subject_name, max_length=32),
    )
def clarification_message(missing_codes: tuple[str, ...]) -> str:
    labels = {
        "STUDENT_NAME": "学生姓名",
        "TIME_RANGE": "年份或学期",
        "EXAM_TYPE": "考试类型（如期中/期末）",
    }
    missing = "、".join(labels[code] for code in missing_codes if code in labels)
    return (
        f"为了准确查询，请补充：{missing}。"
        "示例：查询张三2026年期末考试的成绩。"
    )


def _extract_name(text: str, *, contextual: bool) -> str | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_text(match.group("name"))
    if contextual:
        match = _CONTEXTUAL_NAME_RE.search(text)
        if match:
            return normalize_text(match.group("name"))
    return None


def _extract_year(text: str) -> tuple[int | None, str | None, str | None]:
    match = _YEAR_RE.search(text)
    if not match:
        return None, None, None
    year = int(match.group("year"))
    next_year = match.group("next_year")
    if next_year:
        return year, "ACADEMIC_YEAR", f"{year}-{int(next_year)}"
    if "学年" in text:
        return year, "ACADEMIC_YEAR", f"{year}-{year + 1}"
    return year, "CALENDAR_YEAR", None


def _extract_exam_type(text: str) -> str | None:
    for code, aliases in _EXAM_ALIASES.items():
        if any(alias in text for alias in aliases):
            return code
    return None


def _extract_term_no(text: str) -> int | None:
    for term_no, aliases in _TERM_ALIASES.items():
        if any(alias in text for alias in aliases):
            return term_no
    return None


def _parse_slot_response(response: str) -> ScoreSlots:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        )
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != _SLOT_RESPONSE_KEYS:
        raise ValueError("invalid score slot response shape")

    student_name = _optional_text(payload["studentName"], max_length=32)
    year_value = _optional_int(
        payload["yearValue"],
        allowed=range(2000, 2100),
    )
    year_mode = _optional_enum(payload["yearMode"], _YEAR_MODES)
    academic_year = _optional_academic_year(payload["academicYear"])
    term_no = _optional_int(payload["termNo"], allowed=(1, 2))
    exam_type_code = _optional_enum(
        payload["examTypeCode"],
        _EXAM_TYPE_CODES,
    )
    subject_name = _optional_text(payload["subjectName"], max_length=32)
    return validate_score_slots(
        ScoreSlots(
            student_name=student_name,
            year_value=year_value,
            year_mode=year_mode,
            academic_year=academic_year,
            term_no=term_no,
            exam_type_code=exam_type_code,
            subject_name=subject_name,
        )
    )


def _empty_score_slots() -> ScoreSlots:
    return ScoreSlots(
        student_name=None,
        year_value=None,
        year_mode=None,
        academic_year=None,
        term_no=None,
        exam_type_code=None,
        subject_name=None,
    )


def _merge_score_slots(primary: ScoreSlots, supplement: ScoreSlots) -> ScoreSlots:
    """Fill missing deterministic slots from a validated LLM result."""
    primary_has_year = (
        primary.year_value is not None or primary.academic_year is not None
    )
    if primary_has_year:
        year_value = primary.year_value
        year_mode = primary.year_mode
        academic_year = primary.academic_year
    else:
        year_value = supplement.year_value
        year_mode = supplement.year_mode
        academic_year = supplement.academic_year
    return validate_score_slots(
        ScoreSlots(
            student_name=primary.student_name or supplement.student_name,
            year_value=year_value,
            year_mode=year_mode,
            academic_year=academic_year,
            term_no=primary.term_no or supplement.term_no,
            exam_type_code=primary.exam_type_code or supplement.exam_type_code,
            subject_name=primary.subject_name or supplement.subject_name,
        )
    )


def _optional_text(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("invalid score slot text type")
    normalized = normalize_text(value)
    if not normalized or len(normalized) > max_length:
        raise ValueError("invalid score slot text length")
    return normalized


def _optional_int(
    value: object,
    *,
    allowed: Collection[int],
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        raise ValueError("invalid score slot integer")
    return value


def _optional_enum(value: object, allowed: frozenset[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("invalid score slot enum")
    return value


def _optional_academic_year(value: object) -> str | None:
    academic_year = _optional_text(value, max_length=9)
    if academic_year is None:
        return None
    match = re.fullmatch(r"(20\d{2})-(20\d{2})", academic_year)
    if not match or int(match.group(2)) != int(match.group(1)) + 1:
        raise ValueError("invalid academic year")
    return academic_year


def _message_text(response: Any) -> str:
    content = (
        response
        if isinstance(response, str)
        else getattr(response, "content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, (str, dict))
        )
    return ""
