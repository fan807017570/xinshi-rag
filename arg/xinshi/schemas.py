from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=32000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="当前用户问题")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="本轮之前的对话，按时间顺序；不含当前 message",
    )
    use_rewrite: bool = Field(True, description="是否做指代消解后再做检索同义词扩展")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    rewritten_query: str | None = None
    standalone_query: str | None = Field(
        None,
        description="多轮时指代消解后的检索句（再经同义词扩展后用于 Milvus）",
    )
