from __future__ import annotations

import unittest

from rag.evaluation import EvaluationCase, evaluate_cases


class EvaluationTest(unittest.TestCase):
    def test_metrics_are_calculated_by_stage(self) -> None:
        cases = [
            EvaluationCase(
                question="hit",
                expected_document_ids=("doc_1",),
                expected_answer_terms=("正确",),
            ),
            EvaluationCase(
                question="miss",
                expected_document_ids=("doc_2",),
                expected_answer_terms=("缺失",),
            ),
        ]

        def fake_answer(question: str, **_kwargs):
            if question == "hit":
                return {
                    "answer": "这是正确答案",
                    "retrieval_candidates": [{"document_id": "doc_1"}],
                    "source_details": [{"document_id": "doc_1"}],
                }
            return {
                "answer": "没有命中",
                "retrieval_candidates": [{"document_id": "doc_other"}],
                "source_details": [{"document_id": "doc_other"}],
            }

        report = evaluate_cases(cases, fake_answer)

        self.assertEqual(report["raw_recall_at_20"], 0.5)
        self.assertEqual(report["rerank_hit_at_5"], 0.5)
        self.assertEqual(report["answer_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
