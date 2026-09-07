"""Internal execution context and state for the unified chat graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

from rag.intent_service import ScoreIntent
from rag.score_slot_extractor import ScoreSlots


class ChatChannel(str, Enum):
    """Trusted server-side origin of a chat request."""

    WEB = "WEB"
    WECHAT = "WECHAT"
    INTERNAL = "INTERNAL"
    STREAM = "STREAM"


class ChatResultType(str, Enum):
    """Terminal branch selected by the unified chat graph."""

    RAG = "RAG"
    SCORE_CLARIFICATION = "SCORE_CLARIFICATION"
    SCORE_LINK = "SCORE_LINK"


@dataclass(frozen=True)
class ChatExecutionContext:
    """Trusted metadata supplied by an internal channel adapter."""

    channel: ChatChannel
    score_intent_enabled: bool
    channel_user_id: str | None = None
    message_id: str | None = None
    rag_runtime_validated: bool = False

    @classmethod
    def web(cls) -> ChatExecutionContext:
        """Build context for the synchronous JSON chat endpoint."""
        return cls(channel=ChatChannel.WEB, score_intent_enabled=True)

    @classmethod
    def wechat(
        cls,
        *,
        openid: str,
        message_id: str | None,
    ) -> ChatExecutionContext:
        """Build context from a verified and decrypted WeChat message."""
        return cls(
            channel=ChatChannel.WECHAT,
            score_intent_enabled=True,
            channel_user_id=openid,
            message_id=message_id,
        )

    @classmethod
    def internal(cls) -> ChatExecutionContext:
        """Build backward-compatible context for internal and CLI callers."""
        return cls(channel=ChatChannel.INTERNAL, score_intent_enabled=False)

    @classmethod
    def stream(
        cls,
        *,
        rag_runtime_validated: bool = False,
    ) -> ChatExecutionContext:
        """Build context that preserves the existing SSE-only RAG behavior."""
        return cls(
            channel=ChatChannel.STREAM,
            score_intent_enabled=False,
            rag_runtime_validated=rag_runtime_validated,
        )


class ChatGraphState(TypedDict, total=False):
    """State shared by score-routing and knowledge-RAG graph nodes."""

    user_message: str
    history: list[dict[str, str]]
    use_rewrite: bool
    defer_generation: bool
    channel: ChatChannel
    score_intent_enabled: bool
    channel_user_id: str | None
    message_id: str | None
    rag_runtime_validated: bool
    active_clarification: bool
    intent: ScoreIntent
    score_context_slots: ScoreSlots | None
    score_slots: ScoreSlots
    missing_codes: tuple[str, ...]
    result_type: ChatResultType
    standalone_query: str
    retrieval_query: str
    retrieval: Any
    documents: list[Any]
    prompt: str
    answer: str
