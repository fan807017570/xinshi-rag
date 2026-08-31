"""Inspect and verify Milvus source coverage for the configured collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from pymilvus import Collection, connections

from rag.config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT
from rag.document_metadata import normalize_source_path


@dataclass(frozen=True)
class IndexCoverageReport:
    """Summary of expected and indexed source files."""

    collection: str
    entity_count: int
    expected_source_count: int
    indexed_source_count: int
    coverage_ratio: float
    missing_sources: list[str]
    unexpected_sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def inspect_index_coverage(
    expected_sources: list[str],
    *,
    timeout: float = 10.0,
) -> IndexCoverageReport:
    """Compare expected sources with source metadata stored in Milvus.

    Args:
        expected_sources: Source paths expected in the configured collection.
        timeout: Milvus connection and query timeout in seconds.

    Returns:
        Coverage counts and missing or unexpected source paths.

    Raises:
        pymilvus.exceptions.MilvusException: If Milvus cannot be queried.
    """
    connection_alias = f"xinshi_index_coverage_{uuid4().hex}"
    connections.connect(
        alias=connection_alias,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=timeout,
    )
    try:
        collection = Collection(COLLECTION_NAME, using=connection_alias)
        collection.load(timeout=timeout)
        indexed_sources, indexed_entity_count = _read_indexed_sources(
            collection,
            timeout=timeout,
        )
        expected = {normalize_source_path(source) for source in expected_sources}
        missing_sources = sorted(expected - indexed_sources)
        unexpected_sources = sorted(indexed_sources - expected)
        coverage_ratio = (
            (len(expected) - len(missing_sources)) / len(expected)
            if expected
            else 1.0
        )
        return IndexCoverageReport(
            collection=COLLECTION_NAME,
            entity_count=indexed_entity_count,
            expected_source_count=len(expected),
            indexed_source_count=len(indexed_sources),
            coverage_ratio=round(coverage_ratio, 6),
            missing_sources=missing_sources,
            unexpected_sources=unexpected_sources,
        )
    finally:
        connections.disconnect(connection_alias)


def require_complete_index_coverage(
    expected_sources: list[str],
) -> IndexCoverageReport:
    """Return coverage or raise when expected sources are missing."""
    report = inspect_index_coverage(expected_sources)
    if report.missing_sources:
        raise RuntimeError(
            "Milvus index coverage is incomplete; missing sources: "
            + ", ".join(report.missing_sources)
        )
    return report


def _read_indexed_sources(
    collection: Collection,
    *,
    timeout: float,
) -> tuple[set[str], int]:
    """Read all distinct source values without a fixed query-size limit."""
    sources: set[str] = set()
    entity_count = 0
    iterator = collection.query_iterator(
        batch_size=1000,
        expr="pk >= 0",
        output_fields=["source"],
        timeout=timeout,
    )
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            entity_count += len(rows)
            sources.update(
                normalize_source_path(str(row.get("source") or ""))
                for row in rows
                if row.get("source")
            )
    finally:
        iterator.close()
    return sources, entity_count
