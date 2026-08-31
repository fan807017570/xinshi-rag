from __future__ import annotations

import unittest
from pathlib import Path

from rag.document_metadata import (
    PROJECT_ROOT,
    build_document_id,
    ensure_document_metadata,
    normalize_source_path,
)
from rag.schemas import ChatResponse


class DocumentMetadataTest(unittest.TestCase):
    def test_source_path_is_stable_for_absolute_and_relative_paths(self) -> None:
        relative = Path("rag/data/base/example.md")
        absolute = PROJECT_ROOT / relative

        self.assertEqual(
            normalize_source_path(absolute),
            normalize_source_path(relative),
        )

    def test_document_id_is_stable(self) -> None:
        source = "rag/data/base/example.md"
        first = build_document_id(source, "section", "content")
        second = build_document_id(source, "section", "content")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("doc_"))

    def test_ensure_metadata_preserves_existing_document_id(self) -> None:
        metadata = ensure_document_metadata(
            {
                "document_id": "doc_existing",
                "source": "rag/data/base/example.md",
                "section": "section",
            },
            "content",
        )

        self.assertEqual(metadata["document_id"], "doc_existing")

    def test_chat_response_serializes_structured_source(self) -> None:
        response = ChatResponse(
            answer="answer",
            sources=["section"],
            source_details=[{
                "document_id": "doc_1",
                "source": "rag/data/base/example.md",
                "section": "section",
                "audience_role": "general",
                "retrieve_rank": 1,
                "retrieve_score": 0.9,
                "rerank_rank": 1,
                "rerank_score": 0.8,
            }],
        )

        payload = response.model_dump()

        self.assertEqual(
            payload["source_details"][0]["document_id"],
            "doc_1",
        )


if __name__ == "__main__":
    unittest.main()
