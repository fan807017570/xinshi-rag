"""Stable metadata helpers shared by ingestion and retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_source_path(source: str | Path) -> str:
    """Return a deployment-independent, POSIX-style document source path."""
    source_path = Path(source)
    if source_path.is_absolute():
        try:
            source_path = source_path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            return source_path.as_posix()
    return source_path.as_posix()


def build_document_id(source: str, section: str, content: str) -> str:
    """Build a stable identifier from source, section, and chunk content."""
    normalized_source = normalize_source_path(source)
    material = f"{normalized_source}\0{section}\0{content.strip()}".encode(
        "utf-8",
        errors="ignore",
    )
    return f"doc_{hashlib.sha256(material).hexdigest()[:24]}"


def ensure_document_metadata(
    metadata: dict[str, Any] | None,
    content: str,
) -> dict[str, Any]:
    """Normalize source metadata and add a document identifier when absent."""
    normalized = dict(metadata or {})
    source = normalize_source_path(str(normalized.get("source") or ""))
    section = str(normalized.get("section") or "")
    normalized["source"] = source
    normalized["section"] = section
    normalized.setdefault(
        "document_id",
        build_document_id(source, section, content),
    )
    return normalized
