import logging
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from arg.xinshi.config import *
from arg.xinshi.logutil import configure_logging
from arg.xinshi.md_table import markdown_table_to_readable, split_markdown_tables

configure_logging()
log = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).resolve().parent / "data" / "base"

_TEACHER_KEYWORDS = (
    "老师", "教师", "班主任", "师资", "教职工", "名师", "教学", "校长"
)
_STUDENT_KEYWORDS = (
    "学生", "同学", "学子", "招生", "报名", "班级", "宿舍", "食宿"
)


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
    """先按表格整体切分，再对非表文本做长度切分；表格转为可读中文描述。"""
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


def ingest():
    t0 = time.perf_counter()
    log.info("ingest start docs_dir=%s collection=%s", _DOCS_DIR, COLLECTION_NAME)

    loader = DirectoryLoader(
        str(_DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
    )

    docs = loader.load()
    log.info("loaded %d markdown file(s) in %.3fs", len(docs), time.perf_counter() - t0)
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
    final_docs = []

    for doc in docs:
        md_chunks = md_splitter.split_text(doc.page_content)
        for chunk in md_chunks:
            for sub in _sub_chunks_from_chunk(
                    chunk.page_content, text_splitter, chunk_size
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
                final_docs.append(Document(page_content=sub, metadata=metadata))

    log.info("split into %d chunk(s) in %.3fs", len(final_docs), time.perf_counter() - t0)

    t_emb = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True})
    log.info("embeddings model ready in %.3fs", time.perf_counter() - t_emb)

    t_mv = time.perf_counter()
    Milvus.from_documents(
        final_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        drop_old=True,
        index_params={
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }
    )
    log.info(
        "Milvus.from_documents done in %.3fs total_ingest=%.3fs",
        time.perf_counter() - t_mv,
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    ingest()
