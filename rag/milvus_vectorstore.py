"""Load the Milvus vector store supported by the deployed Milvus 2.4 service."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any


@contextmanager
def _suppress_legacy_adapter_warnings() -> Iterator[None]:
    """Suppress only warnings emitted by the temporary Milvus 2.4 adapter."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`langchain-community` is being sunset.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The class `Milvus` was deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API.*",
            category=UserWarning,
            module=r"pymilvus\.client",
        )
        yield


@lru_cache(maxsize=1)
def _get_milvus_vectorstore_class() -> type[Any]:
    """Return LangChain's Milvus adapter with a clear dependency error.

    The standalone ``langchain-milvus`` releases compatible with LangChain 1.x
    require a newer PyMilvus/Milvus protocol than this project's Milvus 2.4.4
    deployment.  Keep the legacy adapter isolated here until the database is
    upgraded, and suppress only its known legacy-adapter warnings.
    """
    try:
        with _suppress_legacy_adapter_warnings():
            from langchain_community.vectorstores.milvus import Milvus
    except ImportError as exc:
        raise RuntimeError(
            "Milvus 向量存储依赖不可用，请重新安装 rag/requirements-common.txt"
        ) from exc

    return Milvus


def create_milvus_vectorstore(**kwargs: Any) -> Any:
    """Create a vector store while containing legacy-adapter warnings."""
    try:
        with _suppress_legacy_adapter_warnings():
            return _get_milvus_vectorstore_class()(**kwargs)
    except ImportError as exc:
        raise RuntimeError(
            "PyMilvus 不可用，请重新安装 rag/requirements-common.txt"
        ) from exc


def create_milvus_vectorstore_from_documents(
    documents: list[Any],
    embedding: Any,
    **kwargs: Any,
) -> Any:
    """Build a collection from documents using the Milvus 2.4 adapter."""
    try:
        with _suppress_legacy_adapter_warnings():
            return _get_milvus_vectorstore_class().from_documents(
                documents,
                embedding=embedding,
                **kwargs,
            )
    except ImportError as exc:
        raise RuntimeError(
            "PyMilvus 不可用，请重新安装 rag/requirements-common.txt"
        ) from exc
