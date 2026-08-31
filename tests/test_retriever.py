from __future__ import annotations

import unittest
from unittest import mock

from rag import retriever


class RetrieverBackendTest(unittest.TestCase):
    def test_milvus_failure_is_not_downgraded(self) -> None:
        with (
            mock.patch.object(retriever, "USE_MILVUS", True),
            mock.patch.object(
                retriever,
                "_build_milvus_vectorstore",
                side_effect=RuntimeError("unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                retriever.get_vectorstore()

    def test_local_backend_requires_explicit_configuration(self) -> None:
        with (
            mock.patch.object(retriever, "USE_MILVUS", False),
            mock.patch.object(retriever, "_load_documents", return_value=[]),
        ):
            vectorstore = retriever.get_vectorstore()

        self.assertIsInstance(vectorstore, retriever.LocalVectorStore)
        self.assertEqual(
            retriever.get_vectorstore_backend(vectorstore),
            "local",
        )


if __name__ == "__main__":
    unittest.main()
