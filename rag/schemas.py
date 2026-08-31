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
    use_rewrite: bool = Field(
        True,
        description="是否做指代消解后再做检索同义词扩展",
    )
    include_retrieval_trace: bool = Field(
        False,
        description="是否返回原始召回候选，供 Recall@K 评测和问题诊断使用",
    )


class RetrievalSource(BaseModel):
    """Stable identity and ranking information for a retrieved chunk."""

    document_id: str
    source: str
    section: str
    audience_role: Literal["teacher", "student", "general"] = "general"
    retrieve_rank: int | None = None
    retrieve_score: float | None = None
    retrieve_raw_score: float | None = None
    retrieve_metric_type: str | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


class RetrievalMetrics(BaseModel):
    """Counts and timings for each retrieval stage."""

    backend: Literal["milvus", "local"]
    metric_type: str
    query_role: Literal["teacher", "student", "general"]
    requested_top_k: int
    raw_retrieved_count: int
    score_filtered_count: int
    role_filtered_count: int
    rerank_candidate_count: int
    final_count: int
    score_threshold: float
    similarity_search_ms: float
    rerank_ms: float
    total_retrieval_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    source_details: list[RetrievalSource] = Field(default_factory=list)
    retrieval_candidates: list[RetrievalSource] | None = None
    retrieval_metrics: RetrievalMetrics | None = None
    rewritten_query: str | None = None
    standalone_query: str | None = Field(
        None,
        description="多轮时指代消解后的检索句（再经同义词扩展后用于 Milvus）",
    )
