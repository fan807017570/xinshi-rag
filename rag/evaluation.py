"""Evaluate raw retrieval, reranking, and answer quality separately."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RAW_RECALL_K = 20
RERANK_HIT_K = 5


@dataclass(frozen=True)
class EvaluationCase:
    """One labeled RAG evaluation case."""

    question: str
    expected_document_ids: tuple[str, ...]
    expected_answer_terms: tuple[str, ...] = ()
    use_rewrite: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        """Validate and construct a case from one JSON object."""
        question = str(payload.get("question") or "").strip()
        document_ids = tuple(
            str(value).strip()
            for value in payload.get("expected_document_ids", [])
            if str(value).strip()
        )
        answer_terms = tuple(
            str(value).strip()
            for value in payload.get("expected_answer_terms", [])
            if str(value).strip()
        )
        if not question:
            raise ValueError("evaluation case question must not be empty")
        if not document_ids:
            raise ValueError(
                "evaluation case expected_document_ids must not be empty"
            )
        return cls(
            question=question,
            expected_document_ids=document_ids,
            expected_answer_terms=answer_terms,
            use_rewrite=bool(payload.get("use_rewrite", True)),
        )


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load labeled cases from a UTF-8 JSONL file."""
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("case must be a JSON object")
            cases.append(EvaluationCase.from_dict(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid evaluation case at {path}:{line_number}: {exc}"
            ) from exc
    if not cases:
        raise ValueError(f"evaluation file has no cases: {path}")
    return cases


def evaluate_cases(
    cases: list[EvaluationCase],
    answer_function: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate raw Recall@20, rerank Hit@5, and answer term accuracy."""
    results = []
    raw_recall_sum = 0.0
    rerank_hit_sum = 0
    answer_correct_sum = 0
    answer_case_count = 0

    for case in cases:
        output = answer_function(
            case.question,
            history=None,
            use_rewrite=case.use_rewrite,
        )
        expected_ids = set(case.expected_document_ids)
        raw_ids = _document_ids(
            output.get("retrieval_candidates", [])[:RAW_RECALL_K]
        )
        reranked_ids = _document_ids(
            output.get("source_details", [])[:RERANK_HIT_K]
        )
        raw_recall = len(expected_ids & raw_ids) / len(expected_ids)
        rerank_hit = bool(expected_ids & reranked_ids)
        answer_correct = None
        if case.expected_answer_terms:
            answer_case_count += 1
            answer = str(output.get("answer") or "")
            answer_correct = all(
                term in answer for term in case.expected_answer_terms
            )
            answer_correct_sum += int(answer_correct)

        raw_recall_sum += raw_recall
        rerank_hit_sum += int(rerank_hit)
        results.append(
            {
                "question": case.question,
                "raw_recall_at_20": round(raw_recall, 6),
                "rerank_hit_at_5": rerank_hit,
                "answer_correct": answer_correct,
                "raw_document_ids": sorted(raw_ids),
                "reranked_document_ids": sorted(reranked_ids),
            }
        )

    case_count = len(cases)
    return {
        "case_count": case_count,
        "raw_recall_at_20": round(raw_recall_sum / case_count, 6),
        "rerank_hit_at_5": round(rerank_hit_sum / case_count, 6),
        "answer_accuracy": (
            round(answer_correct_sum / answer_case_count, 6)
            if answer_case_count
            else None
        ),
        "answer_case_count": answer_case_count,
        "cases": results,
    }


def _document_ids(source_details: list[dict[str, Any]]) -> set[str]:
    """Extract non-empty document identifiers from serialized sources."""
    return {
        str(source.get("document_id") or "")
        for source in source_details
        if source.get("document_id")
    }


def main() -> None:
    """Run evaluation cases against the configured RAG application."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw Recall@20, rerank Hit@5, and answer accuracy "
            "from a labeled JSONL file."
        )
    )
    parser.add_argument("cases", type=Path, help="Path to labeled JSONL cases")
    args = parser.parse_args()

    from rag.application import answer_question

    report = evaluate_cases(load_evaluation_cases(args.cases), answer_question)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
