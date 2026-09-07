from __future__ import annotations

import importlib
import inspect
import json
import os
import types
import unittest
from unittest import mock

from rag.intent_service import IntentPrediction, ScoreIntent


class _FixedIntentClassifier:
    def __init__(self, probability: float, label: str = "query_score"):
        self._probability = probability
        self._label = label

    def predict(self, _message: str) -> IntentPrediction:
        remaining = max(0.0, 1.0 - self._probability)
        return IntentPrediction(
            predicted_label=self._label,
            query_score_probability=self._probability,
            probabilities={
                "query_score": self._probability,
                "query_teacher": remaining * 0.2,
                "other": remaining * 0.8,
            },
            model_version="test",
        )


class UnifiedChatGraphIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._original_env = {
            name: os.environ.get(name)
            for name in (
                "XINSHI_USE_MILVUS",
                "SCORE_QUERY_ENABLED",
                "SCORE_H5_URL",
                "WECHAT_APPID",
                "WECHAT_CALLBACK_ENABLED",
                "WECHAT_SCORE_CONTEXT_API_URL",
                "WECHAT_SCORE_CONTEXT_SIGNING_SECRET",
            )
        }
        os.environ.update(
            {
                "XINSHI_USE_MILVUS": "false",
                "SCORE_QUERY_ENABLED": "true",
                "SCORE_H5_URL": (
                    "http://example.test/h5/score?appid=wx12345678"
                ),
                "WECHAT_APPID": "wx12345678",
                "WECHAT_CALLBACK_ENABLED": "false",
                "WECHAT_SCORE_CONTEXT_API_URL": "",
                "WECHAT_SCORE_CONTEXT_SIGNING_SECRET": "",
            }
        )
        cls.application = importlib.import_module("rag.application")

    @classmethod
    def tearDownClass(cls) -> None:
        for name, value in cls._original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_web_score_query_executes_score_branch_without_rag_runtime(self) -> None:
        context = importlib.import_module("rag.chat_context")
        slot_llm = mock.Mock()
        slot_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": "张三",
                    "yearValue": 2026,
                    "yearMode": "CALENDAR_YEAR",
                    "academicYear": None,
                    "termNo": None,
                    "examTypeCode": "FINAL",
                    "subjectName": None,
                },
                ensure_ascii=False,
            )
        )
        with mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            side_effect=AssertionError("score branch must bypass the RAG runtime"),
        ), mock.patch.object(
            self.application,
            "retrieve_with_metrics",
            side_effect=AssertionError("score branch must bypass retrieval"),
        ), mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.99),
        ), mock.patch(
            "rag.intent_service.get_intent_llm",
            side_effect=AssertionError("simple score query must bypass intent LLM"),
        ), mock.patch(
            "rag.score_slot_extractor.get_score_slot_llm",
            return_value=slot_llm,
        ):
            result = self.application.answer_question(
                "查询张三2026年期末考试成绩",
                execution_context=context.ChatExecutionContext.web(),
            )

        self.assertIn("/h5/score?appid=wx12345678", result["answer"])
        self.assertIn("studentName=%E5%BC%A0%E4%B8%89", result["answer"])
        self.assertIn("yearValue=2026", result["answer"])
        self.assertIn("examTypeCode=FINAL", result["answer"])
        self.assertEqual([], result["sources"])
        self.assertIsNone(result["retrieval_metrics"])
        slot_llm.invoke.assert_not_called()

    def test_simple_query_with_missing_parameters_is_clarified(self) -> None:
        context = importlib.import_module("rag.chat_context")
        slot_llm = mock.Mock()
        slot_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": None,
                    "yearValue": None,
                    "yearMode": None,
                    "academicYear": None,
                    "termNo": None,
                    "examTypeCode": None,
                    "subjectName": None,
                }
            )
        )
        with mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.99),
        ), mock.patch(
            "rag.intent_service.get_intent_llm",
            side_effect=AssertionError("simple score query must bypass intent LLM"),
        ), mock.patch(
            "rag.score_slot_extractor.get_score_slot_llm",
            return_value=slot_llm,
        ):
            result = self.application.answer_question(
                "查成绩",
                execution_context=context.ChatExecutionContext.web(),
            )

        self.assertTrue(result["answer"].startswith("为了准确查询，请补充："))
        self.assertNotIn("/h5/score", result["answer"])
        self.assertEqual([], result["sources"])
        slot_llm.invoke.assert_called_once()

    def test_web_history_merges_llm_extracted_followup_parameters(self) -> None:
        context = importlib.import_module("rag.chat_context")
        slot_llm = mock.Mock()
        slot_llm.invoke.side_effect = (
            mock.Mock(
                content=json.dumps(
                    {
                        "studentName": None,
                        "yearValue": None,
                        "yearMode": None,
                        "academicYear": None,
                        "termNo": None,
                        "examTypeCode": None,
                        "subjectName": None,
                    }
                )
            ),
            mock.Mock(
                content=json.dumps(
                    {
                        "studentName": "张三",
                        "yearValue": 2026,
                        "yearMode": "CALENDAR_YEAR",
                        "academicYear": None,
                        "termNo": None,
                        "examTypeCode": "FINAL",
                        "subjectName": None,
                    },
                    ensure_ascii=False,
                )
            ),
        )

        with mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.99),
        ), mock.patch(
            "rag.intent_service.get_intent_llm",
            side_effect=AssertionError(
                "contextual parameter reply must skip intent LLM"
            ),
        ), mock.patch(
            "rag.score_slot_extractor.get_score_slot_llm",
            return_value=slot_llm,
        ):
            first = self.application.answer_question(
                "查成绩",
                execution_context=context.ChatExecutionContext.web(),
            )
            history = [
                {"role": "user", "content": "查成绩"},
                {"role": "assistant", "content": first["answer"]},
            ]
            second = self.application.answer_question(
                "张三，2026年期末",
                history=history,
                execution_context=context.ChatExecutionContext.web(),
            )

        self.assertTrue(first["answer"].startswith("为了准确查询，请补充："))
        self.assertIn("studentName=%E5%BC%A0%E4%B8%89", second["answer"])
        self.assertIn("yearValue=2026", second["answer"])
        self.assertEqual(1, slot_llm.invoke.call_count)

    def test_non_fast_web_message_uses_intent_llm(self) -> None:
        context = importlib.import_module("rag.chat_context")
        intent_llm = mock.Mock()
        intent_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "SCORE_QUERY",
                    "confidence": 0.96,
                    "reasonCode": "SCORE_QUERY",
                }
            )
        )
        slot_llm = mock.Mock()
        slot_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": "张三",
                    "yearValue": 2026,
                    "yearMode": "CALENDAR_YEAR",
                    "academicYear": None,
                    "termNo": None,
                    "examTypeCode": "FINAL",
                    "subjectName": None,
                },
                ensure_ascii=False,
            )
        )

        with mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.96),
        ), mock.patch(
            "rag.intent_service.get_intent_llm",
            return_value=intent_llm,
        ), mock.patch(
            "rag.score_slot_extractor.get_score_slot_llm",
            return_value=slot_llm,
        ), mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            side_effect=AssertionError("LLM-classified score query must bypass RAG"),
        ), mock.patch.object(
            self.application,
            "retrieve_with_metrics",
            side_effect=AssertionError(
                "LLM-classified score query must bypass retrieval"
            ),
        ):
            result = self.application.answer_question(
                "想了解张三2026年期末考试的结果",
                execution_context=context.ChatExecutionContext.web(),
            )

        self.assertIn("/h5/score?appid=wx12345678", result["answer"])
        intent_llm.invoke.assert_called_once()
        slot_llm.invoke.assert_not_called()

    def test_internal_caller_keeps_existing_rag_route(self) -> None:
        context = importlib.import_module("rag.chat_context")
        with mock.patch(
            "rag.intent_service.get_intent_llm",
            side_effect=AssertionError("disabled intent routing must bypass LLM"),
        ), mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            side_effect=RuntimeError("rag-route-selected"),
        ), self.assertRaisesRegex(RuntimeError, "rag-route-selected"):
            self.application._chat_graph.invoke(
                self.application._graph_input(
                    "查询张三2026年期末考试成绩",
                    [],
                    True,
                    defer_generation=False,
                    execution_context=context.ChatExecutionContext.internal(),
                )
            )

    def test_general_web_question_preserves_rag_response_shape(self) -> None:
        context = importlib.import_module("rag.chat_context")
        document = types.SimpleNamespace(
            page_content="学校位于测试路。",
            metadata={"section": "学校地址", "source": "base/school.md"},
        )
        retrieval = self.application.RetrievalResult(
            documents=[document],
            raw_documents=[document],
            metrics={"backend": "local"},
        )
        llm = mock.Mock()
        llm.invoke.return_value = "学校位于测试路。"
        intent_llm = mock.Mock()
        intent_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "GENERAL_QUERY",
                    "confidence": 0.99,
                    "reasonCode": "GENERAL_QUERY",
                }
            )
        )

        with mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.0, label="other"),
        ), mock.patch(
            "rag.intent_service.get_intent_llm",
            return_value=intent_llm,
        ), mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            return_value={"llm_backend": "test"},
        ), mock.patch.object(
            self.application,
            "contextualize_for_search",
            return_value="学校在哪里",
        ), mock.patch.object(
            self.application,
            "query_rewrite",
            return_value="学校地址 地理位置",
        ), mock.patch.object(
            self.application,
            "retrieve_with_metrics",
            return_value=retrieval,
        ), mock.patch.object(
            self.application,
            "build_prompt",
            return_value="prompt",
        ), mock.patch.object(
            self.application,
            "get_llm",
            return_value=llm,
        ):
            result = self.application.answer_question(
                "学校在哪里",
                execution_context=context.ChatExecutionContext.web(),
            )

        self.assertEqual("学校位于测试路。", result["answer"])
        self.assertEqual(["学校地址"], result["sources"])
        self.assertEqual("学校地址 地理位置", result["rewritten_query"])
        self.assertEqual("学校在哪里", result["standalone_query"])
        self.assertEqual({"backend": "local"}, result["retrieval_metrics"])
        intent_llm.invoke.assert_not_called()

    def test_prevalidated_stream_context_skips_duplicate_runtime_check(self) -> None:
        with mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            side_effect=AssertionError("runtime validation must not repeat"),
        ):
            update = self.application._validate_rag_runtime_node(
                {"rag_runtime_validated": True}
            )

        context = importlib.import_module("rag.chat_context")
        self.assertEqual(context.ChatResultType.RAG, update["result_type"])

    def test_stream_rag_keeps_meta_delta_done_protocol(self) -> None:
        document = types.SimpleNamespace(
            page_content="学校位于测试路。",
            metadata={"section": "学校地址", "source": "base/school.md"},
        )
        retrieval = self.application.RetrievalResult(
            documents=[document],
            raw_documents=[document],
            metrics={"backend": "local"},
        )
        llm = mock.Mock()
        llm.stream.return_value = iter(("学校位于", "测试路。"))

        with mock.patch(
            "rag.intent_service.get_intent_llm",
            side_effect=AssertionError("stream must bypass score intent LLM"),
        ), mock.patch.object(
            self.application,
            "get_llm_runtime_status",
            side_effect=AssertionError("prevalidated stream must not recheck"),
        ), mock.patch.object(
            self.application,
            "contextualize_for_search",
            return_value="学校在哪里",
        ), mock.patch.object(
            self.application,
            "query_rewrite",
            return_value="学校地址 地理位置",
        ), mock.patch.object(
            self.application,
            "retrieve_with_metrics",
            return_value=retrieval,
        ), mock.patch.object(
            self.application,
            "build_prompt",
            return_value="prompt",
        ), mock.patch.object(
            self.application,
            "get_llm",
            return_value=llm,
        ):
            events = list(
                self.application.stream_answer_question(
                    "学校在哪里",
                    rag_runtime_validated=True,
                )
            )

        self.assertEqual(
            ["meta", "delta", "delta", "done"],
            [event["type"] for event in events],
        )
        self.assertEqual("学校位于测试路。", events[-1]["answer"])

    def test_sync_completion_endpoint_uses_web_score_context(self) -> None:
        from fastapi.testclient import TestClient

        server = importlib.import_module("rag.server")
        self.assertFalse(inspect.iscoroutinefunction(server.create_chat_completion))

        slot_llm = mock.Mock()
        slot_llm.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": "张三",
                    "yearValue": 2026,
                    "yearMode": "CALENDAR_YEAR",
                    "academicYear": None,
                    "termNo": None,
                    "examTypeCode": "FINAL",
                    "subjectName": None,
                },
                ensure_ascii=False,
            )
        )
        with mock.patch(
            "rag.intent_service.get_intent_classifier",
            return_value=_FixedIntentClassifier(0.99),
        ), mock.patch(
            "rag.score_slot_extractor.get_score_slot_llm",
            return_value=slot_llm,
        ), TestClient(server.app) as client:
            response = client.post(
                "/api/chat/completions",
                json={
                    "message": "查询张三2026年期末考试成绩",
                    "openid": "forged-openid",
                    "msg_id": "forged-msg-id",
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertIn("/h5/score?appid=wx12345678", payload["answer"])
        self.assertEqual([], payload["sources"])


if __name__ == "__main__":
    unittest.main()
