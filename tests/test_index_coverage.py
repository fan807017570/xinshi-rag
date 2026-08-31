from __future__ import annotations

import unittest
from unittest import mock

from rag import index_coverage


class IndexCoverageTest(unittest.TestCase):
    def test_entity_count_uses_queried_rows(self) -> None:
        iterator = mock.Mock()
        iterator.next.side_effect = [
            [
                {"source": "rag/data/base/one.md"},
                {"source": "rag/data/base/one.md"},
                {"source": "rag/data/base/two.md"},
            ],
            [],
        ]
        collection = mock.Mock()
        collection.query_iterator.return_value = iterator

        with (
            mock.patch.object(
                index_coverage,
                "Collection",
                return_value=collection,
            ),
            mock.patch.object(index_coverage.connections, "connect"),
            mock.patch.object(index_coverage.connections, "disconnect"),
        ):
            report = index_coverage.inspect_index_coverage([
                "rag/data/base/one.md",
                "rag/data/base/two.md",
            ])

        self.assertEqual(report.entity_count, 3)
        self.assertEqual(report.indexed_source_count, 2)
        self.assertEqual(report.coverage_ratio, 1.0)
        self.assertEqual(report.missing_sources, [])


if __name__ == "__main__":
    unittest.main()
