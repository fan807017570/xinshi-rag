from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from rag.chat_context import ChatExecutionContext, ChatGraphState, ChatResultType
from rag.config import (
    COLLECTION_NAME,
    MAX_HISTORY_MESSAGES,
    ROLE_FILTER_MULTIPLIER,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
)
from rag.document_metadata import ensure_document_metadata
from rag.llm import get_little_llm, get_llm, get_llm_runtime_status
from rag.logutil import configure_logging
from rag.reranker import BGEReranker
from rag.retriever import get_vectorstore, get_vectorstore_backend
from rag.score_query_graph import ScoreQueryWorkflow

configure_logging()

log = logging.getLogger(__name__)

log.info("Loading vectorstore and reranker (cold start may take a while)...")
_t_load = time.perf_counter()
vectorstore = get_vectorstore()
reranker = BGEReranker()
log.info(
    "RAG backends ready in %.2fs (TOP_K_RETRIEVE=%d TOP_K_RERANK=%d)",
    time.perf_counter() - _t_load,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
)


def reload_vectorstore() -> None:
    """ingest 重建索引后调用，刷新内存中的 vectorstore 连接。"""
    global vectorstore
    log.info("reloading vectorstore after reindex...")
    t0 = time.perf_counter()
    vectorstore = get_vectorstore()
    log.info("vectorstore reloaded in %.2fs", time.perf_counter() - t0)


_TEACHER_QUERY_KEYWORDS = ("老师", "教师", "班主任", "师资", "教职工", "名师")
_STUDENT_QUERY_KEYWORDS = ("学生", "同学", "学子", "招生", "报名", "宿舍", "食宿")
_ALUMNI_QUERY_KEYWORDS = ("校友", "杰出校友", "优秀校友", "创业成功", "创业人士", "企业家", "毕业生榜样")
_ALUMNI_REWRITE_TERMS = (
    "杰出校友风采",
    "创业成功人士榜单",
    "程科源",
    "黄亮",
    "翁艇",
    "校友",
    "毕业生",
    "创业",
    "企业家",
)
_RETRIEVE_SCORE_THRESHOLD = 0.2


@dataclass(frozen=True)
class RetrievalResult:
    """Documents and stage-specific diagnostics for one retrieval request."""

    documents: list[Any]
    raw_documents: list[Any]
    metrics: dict[str, Any]


def _infer_query_role(query: str) -> str:
    teacher_hits = sum(1 for kw in _TEACHER_QUERY_KEYWORDS if kw in query)
    student_hits = sum(1 for kw in _STUDENT_QUERY_KEYWORDS if kw in query)
    if teacher_hits > student_hits and teacher_hits > 0:
        return "teacher"
    if student_hits > teacher_hits and student_hits > 0:
        return "student"
    return "general"


def _infer_doc_role(doc) -> str:
    role = (doc.metadata or {}).get("audience_role")
    if role in ("teacher", "student", "general"):
        return role

    text = f"{(doc.metadata or {}).get('section', '')}\n{doc.page_content or ''}"
    teacher_hits = sum(1 for kw in _TEACHER_QUERY_KEYWORDS if kw in text)
    student_hits = sum(1 for kw in _STUDENT_QUERY_KEYWORDS if kw in text)
    if teacher_hits > student_hits and teacher_hits > 0:
        return "teacher"
    if student_hits > teacher_hits and student_hits > 0:
        return "student"
    return "general"


def _retrieve_metric_type() -> str:
    search_params = getattr(vectorstore, "search_params", None) or {}
    metric_type = search_params.get("metric_type")
    if metric_type:
        return str(metric_type).upper()

    get_index = getattr(vectorstore, "_get_index", None)
    if callable(get_index):
        index = get_index() or {}
        metric_type = (index.get("index_param") or {}).get("metric_type")
        if metric_type:
            return str(metric_type).upper()

    return "L2"


def _to_relevance_score(raw_score: float, metric_type: str) -> float:
    if metric_type == "L2":
        return 1.0 / (1.0 + max(raw_score, 0.0))
    return raw_score


def _similarity_search_with_score_filter(
    query: str,
    k: int,
) -> tuple[list[Any], list[Any], str]:
    scored_docs = vectorstore.similarity_search_with_score(query, k=k)
    metric_type = _retrieve_metric_type()
    raw_documents = []
    score_filtered_documents = []
    for rank, (doc, raw_score) in enumerate(scored_docs, start=1):
        raw_score = float(raw_score)
        score = _to_relevance_score(raw_score, metric_type)
        metadata = ensure_document_metadata(doc.metadata, doc.page_content or "")
        metadata["retrieve_rank"] = rank
        metadata["retrieve_score"] = score
        metadata["retrieve_raw_score"] = raw_score
        metadata["retrieve_metric_type"] = metric_type
        doc.metadata = metadata
        raw_documents.append(doc)
        if score >= _RETRIEVE_SCORE_THRESHOLD:
            score_filtered_documents.append(doc)
    return raw_documents, score_filtered_documents, metric_type


def retrieve_with_metrics(query: str) -> RetrievalResult:
    """Retrieve and rerank documents while preserving stage diagnostics."""
    t0 = time.perf_counter()
    query_role = _infer_query_role(query)
    initial_k = TOP_K_RETRIEVE
    if query_role in ("teacher", "student"):
        initial_k = TOP_K_RETRIEVE * ROLE_FILTER_MULTIPLIER

    raw_documents, documents, metric_type = (
        _similarity_search_with_score_filter(query, initial_k)
    )
    score_filtered_count = len(documents)
    t1 = time.perf_counter()
    if query_role in ("teacher", "student"):
        role_docs = [
            document
            for document in documents
            if _infer_doc_role(document) in (query_role, "general")
        ]
        if role_docs:
            documents = role_docs

    role_filtered_count = len(documents)
    rerank_candidate_count = len(documents)
    final_documents = reranker.rerank(query, documents, TOP_K_RERANK)
    t2 = time.perf_counter()
    backend = get_vectorstore_backend(vectorstore)
    metrics = {
        "backend": backend,
        "metric_type": metric_type,
        "query_role": query_role,
        "requested_top_k": initial_k,
        "raw_retrieved_count": len(raw_documents),
        "score_filtered_count": score_filtered_count,
        "role_filtered_count": role_filtered_count,
        "rerank_candidate_count": rerank_candidate_count,
        "final_count": len(final_documents),
        "score_threshold": _RETRIEVE_SCORE_THRESHOLD,
        "similarity_search_ms": round((t1 - t0) * 1000, 3),
        "rerank_ms": round((t2 - t1) * 1000, 3),
        "total_retrieval_ms": round((t2 - t0) * 1000, 3),
    }
    log.info(
        (
            "retrieve: backend=%s role=%s similarity_search=%.3fs rerank=%.3fs "
            "retrieved=%d metric=%s score_threshold=%.2f score_filtered=%d "
            "role_filtered=%d rerank_candidates=%d final_docs=%d"
        ),
        backend,
        query_role,
        t1 - t0,
        t2 - t1,
        len(raw_documents),
        metric_type,
        _RETRIEVE_SCORE_THRESHOLD,
        score_filtered_count,
        role_filtered_count,
        rerank_candidate_count,
        len(final_documents),
    )
    log.debug(
        "retrieve sections: %s",
        [document.metadata.get("section") for document in final_documents],
    )
    return RetrievalResult(
        documents=final_documents,
        raw_documents=raw_documents,
        metrics=metrics,
    )


def retrieve(query: str) -> list[Any]:
    """Return final reranked documents for backward-compatible callers."""
    return retrieve_with_metrics(query).documents


def _source_detail(document: Any) -> dict[str, Any]:
    """Serialize stable source metadata and retrieval scores."""
    metadata = ensure_document_metadata(
        document.metadata,
        document.page_content or "",
    )
    audience_role = metadata.get("audience_role")
    if audience_role not in ("teacher", "student", "general"):
        audience_role = "general"
    return {
        "document_id": metadata["document_id"],
        "source": metadata["source"],
        "section": metadata["section"],
        "audience_role": audience_role,
        "retrieve_rank": metadata.get("retrieve_rank"),
        "retrieve_score": metadata.get("retrieve_score"),
        "retrieve_raw_score": metadata.get("retrieve_raw_score"),
        "retrieve_metric_type": metadata.get("retrieve_metric_type"),
        "rerank_rank": metadata.get("rerank_rank"),
        "rerank_score": metadata.get("rerank_score"),
    }


def get_rag_runtime_status() -> dict[str, Any]:
    """Validate and return observable RAG backend information."""
    status = {
        "retriever_backend": get_vectorstore_backend(vectorstore),
        "collection": COLLECTION_NAME,
        "reranker_backend": reranker.backend,
    }
    status.update(get_llm_runtime_status())
    return status


def _format_history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "家长" if role == "user" else "顾问"
        lines.append(f"{label}：{content}")
    if not lines:
        return ""
    return "近期对话（供理解指代与语气）：\n" + "\n".join(lines) + "\n\n"


def build_prompt(
    user_message: str,
    docs,
    *,
    history: list[dict[str, str]] | None = None,
):
    context = "\n\n".join([
        f"[{d.metadata.get('section')}] {d.page_content}"
        for d in docs
    ])
    hist = _format_history_block(history)
    return f"""
    你是新实中学的招生顾问。回答要亲切自然、有温度、准确、简洁，先直接回答结论。
    要求：
    - 只基于「内部参考」回答，不编造，不夸张承诺。
    - 默认 4-6 句，尽量 250 字以内；只有用户问多个点时才用最多 7 条短列表。
    - 避免啰嗦寒暄和套话；不要说「根据资料显示」「据资料」「综上所述」「从文档中可以看到」。
    - 内部参考没有的信息，直接说「目前资料里没看到」，并建议通过官方渠道确认。
    - 追问上一话题时自然承接，不重复介绍背景。
    内部参考开始：
    {context}
    内部参考结束。
    {hist}用户问题：
    {user_message}
    """


def _message_text(msg) -> str:
    """Chat models return AIMessage; embeddings and prompts need plain str."""
    if isinstance(msg, str):
        return msg
    return getattr(msg, "content", str(msg))


def _chunk_text(chunk) -> str:
    """从 LangChain 的流式 chunk 中提取可展示文本。"""
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts)
    return ""


def contextualize_for_search(
    user_message: str,
    history: list[dict[str, str]] | None,
) -> str:
    """有多轮对话时，把追问改写成可单独检索的完整问句（消解指代）。"""
    text = user_message.strip()
    if not history:
        return text

    lines: list[str] = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "家长" if role == "user" else "顾问"
        lines.append(f"{label}：{content}")
    if not lines:
        return text

    conv = "\n".join(lines)
    llm = get_little_llm()
    prompt = f"""下面是招生咨询对话节选。请把「用户最后一问」改写成一条**独立、完整**的中文句子，用于知识库向量检索。
要求：
- 把「那里、这边、这个、呢、还有吗、同上」等指代补全成具体事物（结合上文）；
- 保留用户真实意图，不要替用户改话题；
- 不要回答内容本身，不要解释。

对话节选：
{conv}

用户最后一问：{text}

只输出改写后的这一句："""
    out = _message_text(llm.invoke(prompt)).strip()
    return out or text


def query_rewrite(query: str) -> str:
    llm = get_llm()
    prompt = f"""
    将用户问题改写成更适合向量检索的查询（可补充同义词，但不要改变主题）。
    规则：
    - 问地址/在哪/校址/位置 → 保留并补充：学校地址、地理位置、位于、校址、广丰 等词。
    - 问面积/多大/占地 → 保留并补充：占地面积、建筑面积、平方米、亩。
    - 问校长/领导/负责人/谁主持 → 保留并补充：校长姓名、学校领导、领导班子、主持人、负责人 等词。
    - 问老师/教师/师资 → 保留并补充：教师团队、师资力量、专兼职教师、教职工 等词。
    - 问费用/学费/收费 → 保留并补充：收费标准、学费、住宿费、费用 等词。
    - 问招生/报名/入学 → 保留并补充：招生简章、报名条件、入学要求、招生政策 等词。
    - 不要改成与原文无关的主题（例如把校长问句改成招生政策）。
    只输出改写后的一句话，不要解释。
    用户问题：{query}
    """
    return _message_text(llm.invoke(prompt))


def _validate_rag_runtime_node(state: ChatGraphState) -> ChatGraphState:
    if not state.get("rag_runtime_validated"):
        get_llm_runtime_status()
    return {"result_type": ChatResultType.RAG}


def _contextualize_node(state: ChatGraphState) -> ChatGraphState:
    started_at = time.perf_counter()
    history = state.get("history") or []
    standalone = contextualize_for_search(
        state["user_message"],
        history if history else None,
    )
    log.info(
        "rag graph contextualize in %.3fs query_len=%d",
        time.perf_counter() - started_at,
        len(standalone),
    )
    return {"standalone_query": standalone}


def _route_after_contextualize(
    state: ChatGraphState,
) -> Literal["rewrite_query", "retrieve"]:
    return "rewrite_query" if state["use_rewrite"] else "retrieve"


def _rewrite_node(state: ChatGraphState) -> ChatGraphState:
    started_at = time.perf_counter()
    rewritten = query_rewrite(state["standalone_query"])
    log.info(
        "rag graph rewrite in %.3fs query_len=%d",
        time.perf_counter() - started_at,
        len(rewritten),
    )
    return {"retrieval_query": rewritten}


def _retrieve_node(state: ChatGraphState) -> ChatGraphState:
    query = state.get("retrieval_query") or state["standalone_query"]
    retrieval = retrieve_with_metrics(query)
    return {
        "retrieval_query": query,
        "retrieval": retrieval,
        "documents": retrieval.documents,
    }


def _build_prompt_node(state: ChatGraphState) -> ChatGraphState:
    return {
        "prompt": build_prompt(
            state["user_message"].strip(),
            state["documents"],
            history=state.get("history") or [],
        )
    }


def _route_after_prompt(
    state: ChatGraphState,
) -> Literal["generate_answer", "end"]:
    return "end" if state.get("defer_generation") else "generate_answer"


def _generate_node(state: ChatGraphState) -> ChatGraphState:
    started_at = time.perf_counter()
    answer = _message_text(get_llm().invoke(state["prompt"]))
    log.info(
        "rag graph generate in %.3fs answer_len=%d",
        time.perf_counter() - started_at,
        len(answer),
    )
    return {"answer": answer}


def _build_chat_graph(score_workflow: ScoreQueryWorkflow):
    builder = StateGraph(ChatGraphState)
    builder.add_node("load_score_context", score_workflow.load_score_context)
    builder.add_node("classify_score_intent", score_workflow.classify_intent)
    builder.add_node("extract_score_slots", score_workflow.extract_slots)
    builder.add_node(
        "merge_or_persist_score_context",
        score_workflow.merge_or_persist_context,
    )
    builder.add_node("build_score_clarification", score_workflow.build_clarification)
    builder.add_node("build_score_h5_reply", score_workflow.build_h5_reply)
    builder.add_node("validate_rag_runtime", _validate_rag_runtime_node)
    builder.add_node("contextualize", _contextualize_node)
    builder.add_node("rewrite_query", _rewrite_node)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("build_prompt", _build_prompt_node)
    builder.add_node("generate_answer", _generate_node)
    builder.add_edge(START, "load_score_context")
    builder.add_edge("load_score_context", "classify_score_intent")
    builder.add_conditional_edges(
        "classify_score_intent",
        score_workflow.route_after_intent,
        {
            "score_query": "extract_score_slots",
            "intent_clarification": "build_score_clarification",
            "rag": "validate_rag_runtime",
        },
    )
    builder.add_edge("extract_score_slots", "merge_or_persist_score_context")
    builder.add_conditional_edges(
        "merge_or_persist_score_context",
        score_workflow.route_after_context,
        {
            "clarify": "build_score_clarification",
            "reply_with_link": "build_score_h5_reply",
        },
    )
    builder.add_edge("build_score_clarification", END)
    builder.add_edge("build_score_h5_reply", END)
    builder.add_edge("validate_rag_runtime", "contextualize")
    builder.add_conditional_edges(
        "contextualize",
        _route_after_contextualize,
        {"rewrite_query": "rewrite_query", "retrieve": "retrieve"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "build_prompt")
    builder.add_conditional_edges(
        "build_prompt",
        _route_after_prompt,
        {"generate_answer": "generate_answer", "end": END},
    )
    builder.add_edge("generate_answer", END)
    return builder.compile(name="xinshi-unified-chat")


_score_query_workflow = ScoreQueryWorkflow.from_env()
_chat_graph = _build_chat_graph(_score_query_workflow)


def _graph_input(
    user_message: str,
    history: list[dict[str, str]] | None,
    use_rewrite: bool,
    *,
    defer_generation: bool,
    execution_context: ChatExecutionContext,
) -> ChatGraphState:
    return {
        "user_message": user_message,
        "history": (history or [])[-MAX_HISTORY_MESSAGES:],
        "use_rewrite": use_rewrite,
        "defer_generation": defer_generation,
        "channel": execution_context.channel,
        "score_intent_enabled": execution_context.score_intent_enabled,
        "channel_user_id": execution_context.channel_user_id,
        "message_id": execution_context.message_id,
        "rag_runtime_validated": execution_context.rag_runtime_validated,
    }


def answer_question(
    user_message: str,
    *,
    history: list[dict[str, str]] | None = None,
    use_rewrite: bool = True,
    execution_context: ChatExecutionContext | None = None,
) -> dict:
    """供 CLI / HTTP 调用：可选多轮 history（不含当前句），检索 + 生成回答。"""
    t_all = time.perf_counter()
    hist = (history or [])[-MAX_HISTORY_MESSAGES:]
    log.info(
        "answer_question start use_rewrite=%s history_len=%d user_len=%d",
        use_rewrite,
        len(hist),
        len(user_message),
    )
    try:
        state = _chat_graph.invoke(
            _graph_input(
                user_message,
                hist,
                use_rewrite,
                defer_generation=False,
                execution_context=(
                    execution_context or ChatExecutionContext.internal()
                ),
            )
        )
        if state["result_type"] is not ChatResultType.RAG:
            return {
                "answer": state["answer"],
                "rewritten_query": None,
                "standalone_query": None,
                "sources": [],
                "source_details": [],
                "retrieval_candidates": None,
                "retrieval_metrics": None,
            }
        retrieval = state["retrieval"]
        docs = state["documents"]
        answer = state["answer"]
        standalone = state["standalone_query"]
        q = state["retrieval_query"]

        sources = [d.metadata.get("section") or "" for d in docs]
        source_details = [_source_detail(document) for document in docs]
        retrieval_candidates = [
            _source_detail(document) for document in retrieval.raw_documents
        ]
        log.info(
            "answer_question ok total=%.3fs sources=%s",
            time.perf_counter() - t_all,
            sources,
        )
        return {
            "answer": answer,
            "rewritten_query": q if use_rewrite else None,
            "standalone_query": standalone,
            "sources": sources,
            "source_details": source_details,
            "retrieval_candidates": retrieval_candidates,
            "retrieval_metrics": retrieval.metrics,
        }
    except Exception:
        log.exception(
            "answer_question failed after %.3fs use_rewrite=%s",
            time.perf_counter() - t_all,
            use_rewrite,
        )
        raise


def stream_answer_question(
    user_message: str,
    *,
    history: list[dict[str, str]] | None = None,
    use_rewrite: bool = True,
    include_retrieval_trace: bool = False,
    rag_runtime_validated: bool = False,
) -> Iterator[dict]:
    """流式版本：先检索，再按 chunk 逐步输出回答文本。"""
    t_all = time.perf_counter()
    hist = history or []
    hist = hist[-MAX_HISTORY_MESSAGES:]
    log.info(
        "stream_answer_question start use_rewrite=%s history_len=%d user_len=%d",
        use_rewrite,
        len(hist),
        len(user_message),
    )

    state = _chat_graph.invoke(
        _graph_input(
            user_message,
            hist,
            use_rewrite,
            defer_generation=True,
            execution_context=ChatExecutionContext.stream(
                rag_runtime_validated=rag_runtime_validated,
            ),
        )
    )
    standalone = state["standalone_query"]
    q = state["retrieval_query"]
    retrieval = state["retrieval"]
    docs = state["documents"]
    sources = [d.metadata.get("section") or "" for d in docs]
    source_details = [_source_detail(document) for document in docs]
    retrieval_candidates = [
        _source_detail(document) for document in retrieval.raw_documents
    ]
    prompt = state["prompt"]
    llm = get_llm()

    meta_event = {
        "type": "meta",
        "sources": sources,
        "source_details": source_details,
        "retrieval_metrics": retrieval.metrics,
        "rewritten_query": q if use_rewrite else None,
        "standalone_query": standalone,
    }
    if include_retrieval_trace:
        meta_event["retrieval_candidates"] = retrieval_candidates
    yield meta_event

    t_llm = time.perf_counter()
    answer_parts: list[str] = []
    for chunk in llm.stream(prompt):
        text = _chunk_text(chunk)
        if not text:
            continue
        answer_parts.append(text)
        yield {"type": "delta", "content": text}

    answer = "".join(answer_parts).strip()
    log.info(
        "stream llm done in %.3fs answer_len=%d total=%.3fs",
        time.perf_counter() - t_llm,
        len(answer),
        time.perf_counter() - t_all,
    )
    yield {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "source_details": source_details,
        "retrieval_metrics": retrieval.metrics,
    }


def main():
    history: list[dict[str, str]] = []
    while True:
        query = input("\n? 问题: ")
        out = answer_question(query, history=history, use_rewrite=True)
        print("\n回答：\n", out["answer"])
        for s in out["sources"]:
            print("-", s)
        history.append({"role": "user", "content": query.strip()})
        history.append({"role": "assistant", "content": out["answer"]})
        history = history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()
