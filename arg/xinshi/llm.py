from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

try:
    from langchain_deepseek import ChatDeepSeek
except Exception:  # pragma: no cover - optional dependency
    ChatDeepSeek = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _env_int(name: str, default: int, min_value: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class _FallbackMessage:
    content: str


class _FallbackChatModel:
    def __init__(self, *, mode: str):
        self.mode = mode

    def invoke(self, prompt: str) -> _FallbackMessage:
        return _FallbackMessage(_render_fallback_response(prompt, self.mode))

    def stream(self, prompt: str):
        text = _render_fallback_response(prompt, self.mode)
        for chunk in _chunk_text(text):
            yield _FallbackMessage(chunk)


def _chunk_text(text: str, chunk_size: int = 32) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for ch in text:
        buf.append(ch)
        size += 1
        if size >= chunk_size or ch in "。！？!?；;\n":
            chunks.append("".join(buf))
            buf = []
            size = 0
    if buf:
        chunks.append("".join(buf))
    return chunks


def _extract_between(prompt: str, start: str, end_markers: tuple[str, ...] = ()) -> str:
    idx = prompt.find(start)
    if idx < 0:
        return ""
    text = prompt[idx + len(start):]
    if end_markers:
        end_positions = [text.find(marker) for marker in end_markers if text.find(marker) >= 0]
        if end_positions:
            text = text[: min(end_positions)]
    return text.strip()


def _extract_query(prompt: str) -> str:
    for marker in ("用户问题：", "用户最后一问：", "当前用户这句话", "问题："):
        value = _extract_between(prompt, marker)
        if value:
            first_line = value.splitlines()[0].strip()
            return first_line.strip("：: \t")
    return ""


def _extract_context(prompt: str) -> str:
    value = _extract_between(prompt, "内部参考：", ("当前用户这句话", "用户问题：", "用户最后一问："))
    return value.strip()


def _expand_query(query: str) -> str:
    extra_terms: list[str] = []
    if any(key in query for key in ("地址", "校址", "位置", "在哪", "哪里")):
        extra_terms.extend(["学校地址", "地理位置", "位于", "校址", "广丰"])
    if any(key in query for key in ("面积", "多大", "占地", "建筑面积", "平方米", "亩")):
        extra_terms.extend(["占地面积", "建筑面积", "平方米", "亩"])
    if any(key in query for key in ("校长", "领导", "负责人", "谁主持")):
        extra_terms.extend(["校长姓名", "学校领导", "领导班子", "负责人"])
    if any(key in query for key in ("老师", "教师", "师资")):
        extra_terms.extend(["教师团队", "师资力量", "专兼职教师", "教职工"])
    if any(key in query for key in ("费用", "学费", "收费")):
        extra_terms.extend(["收费标准", "学费", "住宿费", "费用"])
    if any(key in query for key in ("招生", "报名", "入学")):
        extra_terms.extend(["招生简章", "报名条件", "入学要求", "招生政策"])

    if not extra_terms:
        return query

    seen: set[str] = set()
    deduped = []
    for term in extra_terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return f"{query} {' '.join(deduped)}"


def _sentence_score(query: str, sentence: str) -> int:
    score = 0
    tokens: set[str] = set()
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", query):
        token = match.group(0)
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for idx in range(len(token) - 1):
                tokens.add(token[idx:idx + 2])
    for token in tokens:
        if token and token in sentence:
            score += len(token)
    return score


def _pick_relevant_sentences(query: str, context: str) -> list[str]:
    sentences = [
        line.strip()
        for line in re.split(r"[\n。！？!?；;]+", context)
        if line.strip()
    ]
    if not sentences:
        return []
    ranked = sorted(
        (( _sentence_score(query, sentence), sentence) for sentence in sentences),
        key=lambda item: (item[0], len(item[1])),
        reverse=True,
    )
    picked = [sentence for score, sentence in ranked if score > 0][:2]
    return picked


def _render_fallback_response(prompt: str, mode: str) -> str:
    query = _extract_query(prompt)
    context = _extract_context(prompt)

    if "将用户问题改写成更适合向量检索的查询" in prompt and query:
        return _expand_query(query)

    if "请把「用户最后一问」改写成一条" in prompt and query:
        return query

    if query or context:
        snippets = _pick_relevant_sentences(query or prompt, context)
        if snippets:
            if query:
                return "根据资料，" + "。".join(snippets) + "。"
            return "。".join(snippets) + "。"

    if mode == "rewrite":
        return query or prompt.strip()
    return "目前资料里没看到对应信息，建议通过官方渠道确认。"


def _create_chat_model(*, model_name: str, max_tokens: int, temperature: float, mode: str):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ChatDeepSeek is not None and api_key:
        try:
            return ChatDeepSeek(
                model=model_name,
                api_key=api_key,
                streaming=True,
                stream_usage=True,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            log.warning("DeepSeek model unavailable, falling back to local responder", exc_info=True)
    return _FallbackChatModel(mode=mode)


def get_llm():
    max_tokens = _env_int("DEEPSEEK_MAX_TOKENS", 1024, 128)
    temperature = _env_float("DEEPSEEK_TEMPERATURE", 0.0)
    return _create_chat_model(
        model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        max_tokens=max_tokens,
        temperature=temperature,
        mode="answer",
    )


def get_little_llm():
    max_tokens = _env_int("DEEPSEEK_MAX_TOKENS", 128, 128)
    temperature = _env_float("DEEPSEEK_TEMPERATURE", 0.2)
    return _create_chat_model(
        model_name=os.environ.get("DEEPSEEK_MODEL_LITTLE", "deepseek-v4-flash"),
        max_tokens=max_tokens,
        temperature=temperature,
        mode="rewrite",
    )
