from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from arg.xinshi.config import *
from arg.xinshi.md_table import markdown_table_to_readable, split_markdown_tables

log = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parent
_DOCS_DIRS = (
    _ROOT_DIR / "data" / "base",
    _ROOT_DIR / "data" / "docs",
)
_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+")

try:  # pragma: no cover - optional dependency path
    from langchain_community.vectorstores import Milvus as _Milvus
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except Exception:  # pragma: no cover - optional dependency path
    _Milvus = None  # type: ignore[assignment]
    _HuggingFaceEmbeddings = None  # type: ignore[assignment]


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _infer_audience_role(section: str, content: str) -> str:
    text = f"{section}\n{content}"
    teacher_hits = sum(1 for kw in ("老师", "教师", "班主任", "师资", "教职工", "名师", "教学", "校长") if kw in text)
    student_hits = sum(1 for kw in ("学生", "同学", "学子", "招生", "报名", "班级", "宿舍", "食宿") if kw in text)
    if teacher_hits > student_hits and teacher_hits > 0:
        return "teacher"
    if student_hits > teacher_hits and student_hits > 0:
        return "student"
    return "general"


def _iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in _DOCS_DIRS:
        if root.exists():
            files.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
    return files


def _normalize_block(block: str) -> list[str]:
    pieces: list[str] = []
    for kind, segment in split_markdown_tables(block):
        text = segment.strip()
        if not text:
            continue
        if kind == "table":
            text = markdown_table_to_readable(text)
        for para in re.split(r"\n{2,}", text):
            part = para.strip()
            if part:
                pieces.append(part)
    return pieces


def _split_long_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def _load_documents() -> list[Document]:
    docs: list[Document] = []
    for path in _iter_markdown_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        raw_blocks = [block.strip() for block in re.split(r"\n{2,}", content) if block.strip()]
        if not raw_blocks:
            continue

        rel_path = path.relative_to(_ROOT_DIR)
        for index, block in enumerate(raw_blocks, start=1):
            for sub_index, piece in enumerate(_normalize_block(block), start=1):
                for chunk_index, chunk in enumerate(_split_long_text(piece), start=1):
                    section = f"{path.stem}#{index}.{sub_index}.{chunk_index}"
                    metadata = {
                        "source": str(rel_path),
                        "section": section,
                        "audience_role": _infer_audience_role(section, chunk),
                    }
                    docs.append(Document(page_content=chunk, metadata=metadata))

    log.info("loaded %d local document chunk(s)", len(docs))
    return docs


def _text_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _TERM_RE.finditer(text):
        token = match.group(0).lower()
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
    return terms


def _score_document(query: str, doc: Document) -> float:
    query = query.strip()
    doc_text = doc.page_content.strip()
    if not query or not doc_text:
        return 0.0

    query_terms = Counter(_text_terms(query))
    doc_terms = Counter(_text_terms(doc_text))
    overlap = 0.0
    for term, count in query_terms.items():
        overlap += min(count, doc_terms.get(term, 0)) * max(1, len(term))

    query_weight = sum(count * max(1, len(term)) for term, count in query_terms.items()) or 1.0
    doc_weight = sum(count * max(1, len(term)) for term, count in doc_terms.items()) or 1.0
    term_score = overlap / max(query_weight, doc_weight)

    ratio = SequenceMatcher(None, query[:256], doc_text[:1200]).ratio()
    return max(term_score, ratio * 0.8)


class LocalVectorStore:
    def __init__(self, docs: list[Document]):
        self._docs = docs
        self.search_params = {"metric_type": "L2"}

    def _get_index(self) -> dict[str, Any]:
        return {"index_param": {"metric_type": "L2"}}

    def similarity_search_with_score(self, query: str, k: int = 4):
        scored = []
        for doc in self._docs:
            score = _score_document(query, doc)
            if score <= 0:
                continue
            scored.append((doc, 1.0 - score))

        scored.sort(key=lambda item: item[1])
        return scored[:k]


def _build_milvus_vectorstore():
    if _Milvus is None or _HuggingFaceEmbeddings is None:
        raise RuntimeError("Milvus dependencies are unavailable")

    embeddings = _HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )


def get_vectorstore():
    if os.environ.get("XINSHI_USE_MILVUS", "").strip() == "1":
        try:
            return _build_milvus_vectorstore()
        except Exception:
            log.warning("falling back to local file retriever", exc_info=True)

    return LocalVectorStore(_load_documents())
