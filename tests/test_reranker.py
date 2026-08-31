from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag.reranker import BGEReranker


class RerankerCandidateTest(unittest.TestCase):
    def test_candidate_overflow_fails_instead_of_truncating(self) -> None:
        reranker = BGEReranker.__new__(BGEReranker)
        reranker.candidate_limit = 1
        documents = [
            SimpleNamespace(page_content="one", metadata={}),
            SimpleNamespace(page_content="two", metadata={}),
        ]

        with self.assertRaisesRegex(RuntimeError, "candidate_limit"):
            reranker.rerank("query", documents, 1)


if __name__ == "__main__":
    unittest.main()
