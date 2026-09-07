"""Score-query nodes used by the unified chat graph."""

from __future__ import annotations

import logging
import os
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from rag.chat_context import ChatChannel, ChatGraphState, ChatResultType
from rag.intent_service import (
    ScoreIntent,
    classify_score_intent,
    has_score_safety_veto,
    is_explicit_score_query_rejection,
)
from rag.score_context_client import ScoreContextClient
from rag.score_slot_extractor import (
    ScoreSlots,
    clarification_message,
    extract_score_slots,
    extract_score_slots_hybrid,
    validate_score_slots,
)

log = logging.getLogger(__name__)

_CLARIFICATION_PREFIX = "为了准确查询，请补充："
_INTENT_CLARIFICATION_PREFIX = (
    "你是想查询学生成绩，还是咨询成绩规则？"
)


@dataclass(frozen=True)
class ActiveScoreContext:
    """Active clarification state loaded for one request."""

    active: bool
    slots: ScoreSlots | None = None


@dataclass(frozen=True)
class EffectiveScoreContext:
    """Merged score slots and their remaining requirements."""

    slots: ScoreSlots
    missing_codes: tuple[str, ...]


class ScoreConversationContextProvider(Protocol):
    """Channel-specific score clarification storage contract."""

    def load(
        self,
        *,
        user_id: str | None,
        history: list[dict[str, str]],
    ) -> ActiveScoreContext:
        """Load an active clarification without transport-specific state."""
        raise NotImplementedError

    def merge_and_store(
        self,
        *,
        user_id: str | None,
        message_id: str | None,
        history: list[dict[str, str]],
        previous_slots: ScoreSlots | None,
        incoming_slots: ScoreSlots,
    ) -> EffectiveScoreContext:
        """Merge incoming slots and persist them when the channel supports it."""
        raise NotImplementedError


class WeChatScoreContextProvider:
    """Persist WeChat score clarification through the existing backend API."""

    def __init__(
        self,
        *,
        appid: str,
        context_client: ScoreContextClient,
    ) -> None:
        self._appid = appid
        self._context_client = context_client

    def load(
        self,
        *,
        user_id: str | None,
        history: list[dict[str, str]],
    ) -> ActiveScoreContext:
        del history
        if not user_id:
            return ActiveScoreContext(active=False)
        try:
            active = self._context_client.has_active_clarification(
                appid=self._appid,
                openid=user_id,
            )
        except Exception:  # Infrastructure must not break the ordinary RAG path.
            log.exception("active score clarification lookup failed")
            active = False
        return ActiveScoreContext(active=active)

    def merge_and_store(
        self,
        *,
        user_id: str | None,
        message_id: str | None,
        history: list[dict[str, str]],
        previous_slots: ScoreSlots | None,
        incoming_slots: ScoreSlots,
    ) -> EffectiveScoreContext:
        del history, previous_slots
        if not user_id:
            return EffectiveScoreContext(
                slots=incoming_slots,
                missing_codes=incoming_slots.missing_codes(),
            )
        stored = self._context_client.store(
            appid=self._appid,
            openid=user_id,
            msg_id=message_id,
            slots=incoming_slots,
        )
        effective_slots = stored.slots if stored else incoming_slots
        try:
            effective_slots = validate_score_slots(effective_slots)
        except ValueError:
            log.warning("invalid score slots returned by context backend")
            effective_slots = incoming_slots
        missing_codes = (
            effective_slots.missing_codes()
        )
        return EffectiveScoreContext(
            slots=effective_slots,
            missing_codes=missing_codes,
        )


class WebHistoryScoreContextProvider:
    """Derive score clarification state from client-supplied chat history."""

    def load(
        self,
        *,
        user_id: str | None,
        history: list[dict[str, str]],
    ) -> ActiveScoreContext:
        del user_id
        score_window = _latest_score_clarification_window(history)
        if not score_window:
            return ActiveScoreContext(active=False)
        slots: ScoreSlots | None = None
        for message in score_window:
            extracted = extract_score_slots(message, contextual=slots is not None)
            slots = merge_score_slots(slots, extracted)
        return ActiveScoreContext(active=True, slots=slots)

    def merge_and_store(
        self,
        *,
        user_id: str | None,
        message_id: str | None,
        history: list[dict[str, str]],
        previous_slots: ScoreSlots | None,
        incoming_slots: ScoreSlots,
    ) -> EffectiveScoreContext:
        del user_id, message_id, history
        effective_slots = merge_score_slots(previous_slots, incoming_slots)
        return EffectiveScoreContext(
            slots=effective_slots,
            missing_codes=effective_slots.missing_codes(),
        )


class ScoreQueryWorkflow:
    """Provide score-query nodes for the single unified LangGraph.

    This component deliberately does not build or compile a graph.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        appid: str,
        context_client: ScoreContextClient,
        h5_url_provider: Callable[[], str],
        intent_classifier: Callable[[str], ScoreIntent] | None = None,
        slot_extractor: Callable[[str], ScoreSlots] | None = None,
    ) -> None:
        self._enabled = enabled
        self._h5_url_provider = h5_url_provider
        self._intent_classifier = intent_classifier
        self._slot_extractor = slot_extractor
        self._providers: dict[
            ChatChannel,
            ScoreConversationContextProvider,
        ] = {
            ChatChannel.WECHAT: WeChatScoreContextProvider(
                appid=appid,
                context_client=context_client,
            ),
            ChatChannel.WEB: WebHistoryScoreContextProvider(),
        }

    @classmethod
    def from_env(cls) -> ScoreQueryWorkflow:
        """Build the default workflow while preserving legacy configuration."""
        enabled_value = os.environ.get(
            "SCORE_QUERY_ENABLED",
            os.environ.get("WECHAT_SCORE_QUERY_ENABLED", "false"),
        )
        appid = os.environ.get("WECHAT_APPID", "").strip()
        h5_url = os.environ.get(
            "SCORE_H5_URL",
            os.environ.get("WECHAT_SCORE_H5_URL", ""),
        ).strip()
        return cls(
            enabled=enabled_value.strip().lower() == "true",
            appid=appid,
            context_client=ScoreContextClient(),
            h5_url_provider=lambda: validate_score_h5_url(h5_url, appid),
        )

    def load_score_context(
        self,
        state: ChatGraphState,
    ) -> ChatGraphState:
        """Load active clarification state before final intent routing."""
        if not self._score_enabled(state):
            return {
                "active_clarification": False,
                "score_context_slots": None,
            }
        provider = self._provider_for(state)
        if provider is None:
            return {
                "active_clarification": False,
                "score_context_slots": None,
            }
        context = provider.load(
            user_id=state.get("channel_user_id"),
            history=state.get("history") or [],
        )
        return {
            "active_clarification": context.active,
            "score_context_slots": context.slots,
        }

    def classify_intent(self, state: ChatGraphState) -> ChatGraphState:
        """Resolve contextual replies, then run SVM and optional LLM routing."""
        if not self._score_enabled(state):
            return {"intent": ScoreIntent.GENERAL_QUERY}

        message = state["user_message"]
        if state.get("active_clarification"):
            if has_score_safety_veto(message) or is_explicit_score_query_rejection(
                message
            ):
                return {"intent": ScoreIntent.GENERAL_QUERY}
            if _resolve_clarification_response(message) is ScoreIntent.SCORE_QUERY:
                return {"intent": ScoreIntent.SCORE_QUERY}
            slots = extract_score_slots(message, contextual=True)
            if _contains_slot_supplement(slots):
                return {"intent": ScoreIntent.SCORE_QUERY}

        intent = self._intent_classifier(message) if self._intent_classifier else None
        if intent is None:
            intent = classify_score_intent(message)
        return {"intent": intent}

    def route_after_intent(
        self,
        state: ChatGraphState,
    ) -> Literal["score_query", "intent_clarification", "rag"]:
        """Route score lookups directly to the fixed link response."""
        if self._score_enabled(state) and state["intent"] is ScoreIntent.SCORE_QUERY:
            return "score_query"
        if self._score_enabled(state) and state["intent"] is ScoreIntent.AMBIGUOUS:
            return "intent_clarification"
        return "rag"

    def extract_slots(self, state: ChatGraphState) -> ChatGraphState:
        """Extract score slots locally and use LLM only for missing fields."""
        extraction_text = state["user_message"]
        if state["channel"] is ChatChannel.WEB:
            prior_messages = _latest_score_clarification_window(
                state.get("history") or []
            )
            if prior_messages:
                extraction_text = "\n".join(
                    [*prior_messages, state["user_message"]]
                )
        slots = (
            self._slot_extractor(extraction_text)
            if self._slot_extractor is not None
            else extract_score_slots_hybrid(
                extraction_text,
                contextual=True,
            )
        )
        return {"score_slots": slots}

    def merge_or_persist_context(self, state: ChatGraphState) -> ChatGraphState:
        """Merge score slots through the trusted channel-specific provider."""
        incoming_slots = state["score_slots"]
        provider = self._provider_for(state)
        if provider is None:
            effective = EffectiveScoreContext(
                slots=incoming_slots,
                missing_codes=incoming_slots.missing_codes(),
            )
        else:
            effective = provider.merge_and_store(
                user_id=state.get("channel_user_id"),
                message_id=state.get("message_id"),
                history=state.get("history") or [],
                previous_slots=state.get("score_context_slots"),
                incoming_slots=incoming_slots,
            )
        return {
            "score_slots": effective.slots,
            "missing_codes": effective.missing_codes,
        }

    def route_after_context(
        self,
        state: ChatGraphState,
    ) -> Literal["clarify", "reply_with_link"]:
        """Route an incomplete lookup to clarification, otherwise to H5."""
        return "clarify" if state["missing_codes"] else "reply_with_link"

    def build_clarification(self, state: ChatGraphState) -> ChatGraphState:
        """Build the existing deterministic missing-slot response."""
        if state.get("intent") is ScoreIntent.AMBIGUOUS:
            return {
                "answer": _INTENT_CLARIFICATION_PREFIX,
                "result_type": ChatResultType.SCORE_CLARIFICATION,
            }
        return {
            "answer": clarification_message(state["missing_codes"]),
            "result_type": ChatResultType.SCORE_CLARIFICATION,
        }

    def build_h5_reply(self, state: ChatGraphState) -> ChatGraphState:
        """Build a score-query URL from validated, URL-encoded slot values."""
        url = append_score_query_parameters(
            self._h5_url_provider(),
            state["score_slots"],
        )
        return {
            "answer": (
                "已为你找到学生成绩查询入口，请在微信内打开。\n"
                + url
            ),
            "result_type": ChatResultType.SCORE_LINK,
        }

    def _score_enabled(self, state: ChatGraphState) -> bool:
        return self._enabled and state.get("score_intent_enabled") is True

    def _provider_for(
        self,
        state: ChatGraphState,
    ) -> ScoreConversationContextProvider | None:
        return self._providers.get(state["channel"])


def validate_score_h5_url(url: str, appid: str) -> str:
    """Validate that a score reply uses only the fixed configured H5 URL."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if parsed.scheme != "http" or not parsed.netloc or parsed.path != "/h5/score":
        raise RuntimeError("SCORE_H5_URL must be a fixed HTTP /h5/score URL")
    if query.get("appid") != [appid] or set(query) != {"appid"}:
        raise RuntimeError("SCORE_H5_URL must contain only the configured appid")
    return url


def append_score_query_parameters(url: str, slots: ScoreSlots) -> str:
    """Append non-empty validated score slots to the fixed H5 URL."""
    slots = validate_score_slots(slots)
    parsed = urllib.parse.urlparse(url)
    query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_items.extend(
        (key, str(value))
        for key, value in slots.to_payload().items()
        if value is not None
    )
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query_items))
    )


def merge_score_slots(
    previous: ScoreSlots | None,
    incoming: ScoreSlots,
) -> ScoreSlots:
    """Merge a new slot extraction over a previous clarification context."""
    if previous is None:
        return incoming
    return ScoreSlots(
        student_name=incoming.student_name or previous.student_name,
        year_value=incoming.year_value or previous.year_value,
        year_mode=incoming.year_mode or previous.year_mode,
        academic_year=incoming.academic_year or previous.academic_year,
        term_no=incoming.term_no or previous.term_no,
        exam_type_code=incoming.exam_type_code or previous.exam_type_code,
        subject_name=incoming.subject_name or previous.subject_name,
    )


def _latest_score_clarification_window(
    history: list[dict[str, str]],
) -> list[str]:
    if not history:
        return []
    last_message = history[-1]
    last_content = last_message.get("content") or ""
    if last_message.get("role") != "assistant" or not (
        last_content.startswith(_CLARIFICATION_PREFIX)
        or last_content.startswith(_INTENT_CLARIFICATION_PREFIX)
    ):
        return []

    user_messages: list[str] = []
    for message in reversed(history):
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "assistant" and not (
            content.startswith(_CLARIFICATION_PREFIX)
            or content.startswith(_INTENT_CLARIFICATION_PREFIX)
        ):
            break
        if role == "user" and content:
            user_messages.append(content)
    user_messages.reverse()
    if not user_messages:
        return []
    # The assistant prefix is generated by this workflow, so it is already a
    # trusted marker that the preceding user messages belong to score routing.
    # Do not re-run the SVM here: that would duplicate inference and could make
    # history loading depend on the classifier artifact before current-message
    # routing has a chance to fail closed.
    return user_messages


def _resolve_clarification_response(message: str) -> ScoreIntent | None:
    """Resolve only an affirmative score-query clarification by simple match."""
    text = "".join((message or "").split()).casefold()
    if text in {
        "是",
        "是的",
        "对",
        "对的",
        "查一下",
        "查成绩",
        "看分数",
        "查分",
    }:
        return ScoreIntent.SCORE_QUERY
    return None


def _contains_slot_supplement(slots: ScoreSlots) -> bool:
    return any(
        (
            slots.student_name,
            slots.year_value,
            slots.academic_year,
            slots.term_no,
            slots.exam_type_code,
            slots.subject_name,
        )
    )
