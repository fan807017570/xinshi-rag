import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
)
from rag.document_metadata import ensure_document_metadata, normalize_source_path
from rag.index_coverage import require_complete_index_coverage
from rag.logutil import configure_logging
from rag.md_table import markdown_table_to_readable, split_markdown_tables
from rag.milvus_vectorstore import create_milvus_vectorstore_from_documents

configure_logging()
log = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).resolve().parent / "data" / "base"

_TEACHER_KEYWORDS = (
    "老师", "教师", "班主任", "师资", "教职工", "名师", "教学", "校长"
)
_STUDENT_KEYWORDS = (
    "学生", "同学", "学子", "招生", "报名", "班级", "宿舍", "食宿"
)


def list_ingest_source_paths() -> list[str]:
    """Return source paths currently eligible for Milvus ingestion."""
    if not _DOCS_DIR.exists():
        return []
    return [
        normalize_source_path(path)
        for path in sorted(_DOCS_DIR.rglob("*.md"))
        if path.is_file()
    ]


def _load_markdown_documents() -> list[Document]:
    """Read Markdown files without the deprecated community loader package."""
    documents: list[Document] = []
    if not _DOCS_DIR.exists():
        return documents

    for source in list_ingest_source_paths():
        path = Path(__file__).resolve().parent.parent / source
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            log.warning("skip unreadable markdown file path=%s error=%s", path, exc)
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": source},
            )
        )
    return documents


def _infer_audience_role(section: str, content: str) -> str:
    """给 chunk 打粗粒度角色标签，便于检索阶段按意图过滤。"""
    text = f"{section}\n{content}"
    teacher_hits = sum(1 for kw in _TEACHER_KEYWORDS if kw in text)
    student_hits = sum(1 for kw in _STUDENT_KEYWORDS if kw in text)
    if teacher_hits > student_hits and teacher_hits > 0:
        return "teacher"
    if student_hits > teacher_hits and student_hits > 0:
        return "student"
    return "general"


def _sub_chunks_from_chunk(
    chunk_content: str,
    text_splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[str]:
    """保留表格整体结构，再对普通文本做长度切分。"""
    pieces: list[str] = []
    for kind, segment in split_markdown_tables(chunk_content):
        seg = segment.strip()
        if not seg:
            continue
        if kind == "table":
            readable = markdown_table_to_readable(seg)
            if len(readable) > chunk_size * 2:
                pieces.extend(text_splitter.split_text(readable))
            else:
                pieces.append(readable)
        else:
            pieces.extend(text_splitter.split_text(seg))
    return pieces


def ingest() -> dict[str, Any]:
    """Rebuild the configured Milvus collection and verify source coverage."""
    t0 = time.perf_counter()
    log.info("ingest start docs_dir=%s collection=%s", _DOCS_DIR, COLLECTION_NAME)

    docs = _load_markdown_documents()
    log.info(
        "loaded %d markdown file(s) in %.3fs",
        len(docs),
        time.perf_counter() - t0,
    )
    if not docs:
        raise RuntimeError(f"未在 {_DOCS_DIR} 中找到可导入的 Markdown 文件")
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers)
    chunk_size = 256
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=80,
    )
    final_docs: list[Document] = []

    for doc in docs:
        md_chunks = md_splitter.split_text(doc.page_content)
        for chunk in md_chunks:
            for sub in _sub_chunks_from_chunk(
                chunk.page_content,
                text_splitter,
                chunk_size,
            ):
                if not sub.strip():
                    continue
                metadata = chunk.metadata.copy()
                for hdr in ("h1", "h2", "h3"):
                    metadata.setdefault(hdr, "")
                metadata["source"] = doc.metadata.get("source", "")
                metadata["section"] = ">".join(
                    [
                        metadata.get("h1", ""),
                        metadata.get("h2", ""),
                        metadata.get("h3", ""),
                    ]
                )
                metadata["audience_role"] = _infer_audience_role(
                    metadata["section"], sub
                )
                metadata = ensure_document_metadata(metadata, sub)
                final_docs.append(Document(page_content=sub, metadata=metadata))

    log.info(
        "split into %d chunk(s) in %.3fs",
        len(final_docs),
        time.perf_counter() - t0,
    )

    t_emb = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("embeddings model ready in %.3fs", time.perf_counter() - t_emb)

    t_mv = time.perf_counter()
    create_milvus_vectorstore_from_documents(
        final_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        drop_old=True,
        index_params={
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        },
    )
    expected_sources = sorted(
        {
            str(doc.metadata.get("source") or "")
            for doc in docs
            if doc.metadata.get("source")
        }
    )
    coverage = require_complete_index_coverage(expected_sources)
    log.info(
        (
            "Milvus.from_documents done in %.3fs total_ingest=%.3fs "
            "entities=%d source_coverage=%d/%d"
        ),
        time.perf_counter() - t_mv,
        time.perf_counter() - t0,
        coverage.entity_count,
        coverage.indexed_source_count,
        coverage.expected_source_count,
    )
    return coverage.to_dict()


if __name__ == "__main__":
    ingest()
