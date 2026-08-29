from __future__ import annotations

import logging
import time
from typing import Iterator

from arg.xinshi.logutil import configure_logging, preview_text

configure_logging()

from arg.xinshi.config import (
    MAX_HISTORY_MESSAGES,
    ROLE_FILTER_MULTIPLIER,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
)
from arg.xinshi.llm import get_llm, get_little_llm
from arg.xinshi.reranker import BGEReranker
from arg.xinshi.retriever import get_vectorstore

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


def _similarity_search_with_score_filter(query: str, k: int):
    scored_docs = vectorstore.similarity_search_with_score(query, k=k)
    metric_type = _retrieve_metric_type()
    docs = []
    for doc, raw_score in scored_docs:
        raw_score = float(raw_score)
        score = _to_relevance_score(raw_score, metric_type)
        metadata = dict(doc.metadata or {})
        metadata["retrieve_score"] = score
        metadata["retrieve_raw_score"] = raw_score
        metadata["retrieve_metric_type"] = metric_type
        doc.metadata = metadata
        if score >= _RETRIEVE_SCORE_THRESHOLD:
            docs.append(doc)
    return docs, len(scored_docs), metric_type


def retrieve(query: str):
    t0 = time.perf_counter()
    query_role = _infer_query_role(query)
    initial_k = TOP_K_RETRIEVE
    if query_role in ("teacher", "student"):
        initial_k = TOP_K_RETRIEVE * ROLE_FILTER_MULTIPLIER

    docs, retrieved_count, metric_type = _similarity_search_with_score_filter(
        query, initial_k
    )
    score_filtered_count = len(docs)
    t1 = time.perf_counter()
    if query_role in ("teacher", "student"):
        role_docs = [
            d for d in docs if _infer_doc_role(d) in (query_role, "general")
        ]
        if role_docs:
            docs = role_docs

    filtered_count = len(docs)
    docs = reranker.rerank(query, docs, TOP_K_RERANK)
    t2 = time.perf_counter()
    log.info(
        (
            "retrieve: role=%s similarity_search=%.3fs rerank=%.3fs "
            "retrieved=%d metric=%s score_threshold=%.2f score_filtered=%d "
            "role_filtered=%d final_docs=%d"
        ),
        query_role,
        t1 - t0,
        t2 - t1,
        retrieved_count,
        metric_type,
        _RETRIEVE_SCORE_THRESHOLD,
        score_filtered_count,
        filtered_count,
        len(docs),
    )
    log.debug(
        "retrieve sections: %s",
        [d.metadata.get("section") for d in docs],
    )
    return docs


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
    内部参考：
    {context}
    {hist}当前用户这句话（请结合上文理解「那里、这个、还有吗」等指代）：
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


def answer_question(
    user_message: str,
    *,
    history: list[dict[str, str]] | None = None,
    use_rewrite: bool = True,
) -> dict:
    """供 CLI / HTTP 调用：可选多轮 history（不含当前句），检索 + 生成回答。"""
    t_all = time.perf_counter()
    hist = history or []
    hist = hist[-MAX_HISTORY_MESSAGES:]
    log.info(
        "answer_question start use_rewrite=%s history_len=%d user_len=%d preview=%r",
        use_rewrite,
        len(hist),
        len(user_message),
        preview_text(user_message, 100),
    )
    try:
        t_ctx = time.perf_counter()
        standalone = contextualize_for_search(user_message, hist if hist else None)
        log.info(
            "contextualize_for_search in %.3fs preview=%r",
            time.perf_counter() - t_ctx,
            preview_text(standalone, 100),
        )

        if use_rewrite:
            t_rw = time.perf_counter()
            q = query_rewrite(standalone)
            log.info(
                "query_rewrite done in %.3fs preview=%r",
                time.perf_counter() - t_rw,
                preview_text(q, 100),
            )
        else:
            q = standalone

        docs = retrieve(q)

        t_llm = time.perf_counter()
        prompt = build_prompt(user_message.strip(), docs, history=hist)
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = _message_text(response)
        log.info(
            "llm invoke done in %.3fs answer_len=%d",
            time.perf_counter() - t_llm,
            len(answer),
        )

        sources = [d.metadata.get("section") or "" for d in docs]
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
) -> Iterator[dict]:
    """流式版本：先检索，再按 chunk 逐步输出回答文本。"""
    t_all = time.perf_counter   ()
    hist = history or []
    hist = hist[-MAX_HISTORY_MESSAGES:]
    log.info(
        "stream_answer_question start use_rewrite=%s history_len=%d user_len=%d preview=%r",
        use_rewrite,
        len(hist),
        len(user_message),
        preview_text(user_message, 100),
    )

    t_ctx = time.perf_counter()
    standalone = contextualize_for_search(user_message, hist if hist else None)
    log.info(
        "stream contextualize_for_search in %.3fs preview=%r",
        time.perf_counter() - t_ctx,
        preview_text(standalone, 100),
    )

    if use_rewrite:
        t_rw = time.perf_counter()
        q = query_rewrite(standalone)
        log.info(
            "stream query_rewrite done in %.3fs preview=%r",
            time.perf_counter() - t_rw,
            preview_text(q, 100),
        )
    else:
        q = standalone

    docs = retrieve(q)
    sources = [d.metadata.get("section") or "" for d in docs]
    prompt = build_prompt(user_message.strip(), docs, history=hist)
    llm = get_llm()

    yield {
        "type": "meta",
        "sources": sources,
        "rewritten_query": q if use_rewrite else None,
        "standalone_query": standalone,
    }

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
    yield {"type": "done", "answer": answer, "sources": sources}


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
