from __future__ import annotations

import hashlib
import hmac
import json
import os
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

from rag.chat_context import ChatChannel, ChatGraphState, ChatResultType
from rag.intent_service import (
    IntentModelError,
    IntentPrediction,
    LocalSvmIntentClassifier,
    ScoreIntent,
    classify_score_intent,
    classify_score_intent_from_prediction,
    classify_score_intent_with_llm,
)
from rag.score_context_client import ScoreContextClient
from rag.score_query_graph import ScoreQueryWorkflow, append_score_query_parameters
from rag.score_slot_extractor import (
    ScoreSlots,
    clarification_message,
    extract_score_slots,
    extract_score_slots_hybrid,
    extract_score_slots_with_llm,
)
from rag.wechat_crypto import WeChatConfig, WeChatCryptoError
from rag.wechat_service import build_encrypted_chat_reply


class _FakeEncoder:
    def encode(self, messages):
        return [[0.0] for _ in messages]


class _FakeSvm:
    classes_ = ("other", "query_score", "query_teacher")

    def __init__(self, *, query_score_probability, predicted_label="query_score"):
        self.query_score_probability = query_score_probability
        self.predicted_label = predicted_label

    def predict(self, _embeddings):
        return [self.predicted_label]

    def predict_proba(self, _embeddings):
        query_score = self.query_score_probability
        remaining = max(0.0, 1.0 - query_score)
        return [[remaining * 0.8, query_score, remaining * 0.2]]


def _fake_intent_classifier(
    query_score_probability=0.99,
    predicted_label="query_score",
):
    return LocalSvmIntentClassifier(
        encoder=_FakeEncoder(),
        classifier=_FakeSvm(
            query_score_probability=query_score_probability,
            predicted_label=predicted_label,
        ),
    )


def _workflow_intent_classifier(message: str) -> ScoreIntent:
    text = "".join((message or "").split())
    if "查成绩" in text or "查询" in text and "成绩" in text:
        return ScoreIntent.SCORE_QUERY
    return ScoreIntent.GENERAL_QUERY


class ScoreQueryIntentTest(unittest.TestCase):
    def test_complete_score_query(self) -> None:
        slots = extract_score_slots("查询张三，2026年期末考试的成绩")

        self.assertEqual(
            ScoreIntent.SCORE_QUERY,
            classify_score_intent(
                "查询张三2026年期末考试的成绩",
                classifier=_fake_intent_classifier(),
            ),
        )
        self.assertEqual("张三", slots.student_name)
        self.assertEqual(2026, slots.year_value)
        self.assertEqual("CALENDAR_YEAR", slots.year_mode)
        self.assertIsNone(slots.term_no)
        self.assertEqual("FINAL", slots.exam_type_code)
        self.assertEqual((), slots.missing_codes())

    def test_svm_high_confidence_bypasses_llm(self) -> None:
        model = mock.Mock()

        result = classify_score_intent(
            "查询张三2026年期末考试成绩",
            classifier=_fake_intent_classifier(0.98),
            llm_model=model,
        )

        self.assertEqual(ScoreIntent.SCORE_QUERY, result)
        model.invoke.assert_not_called()

    def test_svm_middle_band_uses_llm_second_pass(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "SCORE_QUERY",
                    "confidence": 0.91,
                    "reasonCode": "SCORE_OBJECT_WITH_QUERY_ACTION",
                }
            )
        )

        result = classify_score_intent(
            "张三这次考试怎么样",
            classifier=_fake_intent_classifier(0.90),
            llm_model=model,
        )

        self.assertEqual(ScoreIntent.SCORE_QUERY, result)
        model.invoke.assert_called_once()

    def test_svm_low_confidence_routes_to_general_without_llm(self) -> None:
        model = mock.Mock()

        result = classify_score_intent(
            "学校什么时候开学",
            classifier=_fake_intent_classifier(0.50),
            llm_model=model,
        )

        self.assertEqual(ScoreIntent.GENERAL_QUERY, result)
        model.invoke.assert_not_called()

    def test_non_score_categories_are_kept_for_business_routing(self) -> None:
        cases = (
            ("学校成绩制度是什么", ScoreIntent.SCORE_POLICY),
            ("老师怎么录入成绩", ScoreIntent.SCORE_OPERATION),
            ("学校什么时候开学", ScoreIntent.GENERAL_QUERY),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    expected,
                    classify_score_intent(
                        message,
                        classifier=_fake_intent_classifier(
                            0.05,
                            predicted_label="other",
                        ),
                    ),
                )

    def test_explicit_rejection_is_safe_even_with_high_svm_score(self) -> None:
        model = mock.Mock()

        result = classify_score_intent(
            "我不需要成绩查询",
            classifier=_fake_intent_classifier(0.99),
            llm_model=model,
        )

        self.assertEqual(ScoreIntent.GENERAL_QUERY, result)
        model.invoke.assert_not_called()

    def test_local_classifier_validates_model_output(self) -> None:
        classifier = LocalSvmIntentClassifier(
            encoder=_FakeEncoder(),
            classifier=mock.Mock(
                classes_=("other", "query_teacher"),
                predict=mock.Mock(return_value=["other"]),
                predict_proba=mock.Mock(return_value=[[1.0, 0.0]]),
            ),
        )

        with self.assertRaises(IntentModelError):
            classifier.predict("查成绩")

    def test_configurable_svm_thresholds_select_middle_band(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "SCORE_QUERY",
                    "confidence": 0.90,
                    "reasonCode": "SCORE_QUERY",
                }
            )
        )
        prediction = IntentPrediction(
            predicted_label="query_score",
            query_score_probability=0.80,
            probabilities={
                "query_score": 0.80,
                "query_teacher": 0.10,
                "other": 0.10,
            },
            model_version="test",
        )

        with mock.patch.dict(
            os.environ,
            {
                "SCORE_INTENT_THRESHOLD": "0.85",
                "SCORE_INTENT_AMBIGUOUS_THRESHOLD": "0.75",
                "SCORE_INTENT_LLM_THRESHOLD": "0.80",
            },
            clear=True,
        ):
            result = classify_score_intent_from_prediction(
                "张三这次考试怎么样",
                prediction,
                llm_model=model,
            )

        self.assertEqual(ScoreIntent.SCORE_QUERY, result)
        model.invoke.assert_called_once()

    def test_missing_slots_have_fixed_clarification(self) -> None:
        slots = extract_score_slots("查成绩")

        self.assertEqual(
            ("STUDENT_NAME", "TIME_RANGE", "EXAM_TYPE"),
            slots.missing_codes(),
        )
        self.assertIn("学生姓名", clarification_message(slots.missing_codes()))

    def test_hybrid_slot_extraction_skips_llm_for_complete_text(self) -> None:
        model = mock.Mock()

        for message in (
            "查询张三2026年期末考试成绩",
            "张三的2026年期末考试成绩",
            "看一下张三2026年期末数学成绩",
        ):
            with self.subTest(message=message):
                slots = extract_score_slots_hybrid(
                    message,
                    contextual=True,
                    model=model,
                )

                self.assertEqual("张三", slots.student_name)
                self.assertEqual(2026, slots.year_value)
                self.assertEqual("FINAL", slots.exam_type_code)
        model.invoke.assert_not_called()

    def test_hybrid_slot_extraction_preserves_rules_and_fills_missing_fields(
        self,
    ) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": "李四",
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

        slots = extract_score_slots_hybrid("查询张三成绩", model=model)

        self.assertEqual("张三", slots.student_name)
        self.assertEqual(2026, slots.year_value)
        self.assertEqual("FINAL", slots.exam_type_code)
        model.invoke.assert_called_once()

    def test_hybrid_slot_extraction_keeps_partial_rules_when_llm_fails(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(content="not-json")

        slots = extract_score_slots_hybrid("查询张三成绩", model=model)

        self.assertEqual("张三", slots.student_name)
        self.assertEqual(("TIME_RANGE", "EXAM_TYPE"), slots.missing_codes())

    def test_hybrid_slot_extraction_supports_contextual_name_in_history(self) -> None:
        model = mock.Mock()

        slots = extract_score_slots_hybrid(
            "查成绩\n张三，2026年期末",
            contextual=True,
            model=model,
        )

        self.assertEqual("张三", slots.student_name)
        self.assertEqual(2026, slots.year_value)
        self.assertEqual("FINAL", slots.exam_type_code)
        model.invoke.assert_not_called()

    def test_term_number_is_a_valid_time_range(self) -> None:
        slots = extract_score_slots("张三，上学期期末", contextual=True)

        self.assertEqual("张三", slots.student_name)
        self.assertEqual(1, slots.term_no)
        self.assertEqual("FINAL", slots.exam_type_code)
        self.assertEqual((), slots.missing_codes())
        self.assertEqual(1, slots.to_payload()["termNo"])

    def test_llm_extracts_strict_score_query_parameters(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "studentName": "张三",
                    "yearValue": 2026,
                    "yearMode": "CALENDAR_YEAR",
                    "academicYear": None,
                    "termNo": None,
                    "examTypeCode": "FINAL",
                    "subjectName": "数学",
                },
                ensure_ascii=False,
            )
        )

        slots = extract_score_slots_with_llm(
            "查询张三2026年期末数学成绩",
            model=model,
        )

        self.assertEqual("张三", slots.student_name)
        self.assertEqual(2026, slots.year_value)
        self.assertEqual("FINAL", slots.exam_type_code)
        self.assertEqual("数学", slots.subject_name)
        self.assertEqual((), slots.missing_codes())
        model.invoke.assert_called_once()

    def test_invalid_slot_llm_output_returns_all_required_fields_missing(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(content='{"studentName":"张三"}')

        slots = extract_score_slots_with_llm("查询张三成绩", model=model)

        self.assertEqual(
            ("STUDENT_NAME", "TIME_RANGE", "EXAM_TYPE"),
            slots.missing_codes(),
        )

    def test_score_parameters_are_url_encoded_and_appended(self) -> None:
        slots = ScoreSlots(
            student_name="张 三&同学",
            year_value=2026,
            year_mode="CALENDAR_YEAR",
            academic_year=None,
            term_no=None,
            exam_type_code="FINAL",
            subject_name="数学",
        )

        url = append_score_query_parameters(
            "http://example.test/h5/score?appid=wx12345678",
            slots,
        )
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)

        self.assertEqual(["wx12345678"], query["appid"])
        self.assertEqual(["张 三&同学"], query["studentName"])
        self.assertEqual(["2026"], query["yearValue"])
        self.assertEqual(["FINAL"], query["examTypeCode"])

    def test_non_fast_path_is_sent_to_strict_llm_classifier(self) -> None:
        model = mock.Mock()
        model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "SCORE_QUERY",
                    "confidence": 0.91,
                    "reasonCode": "SCORE_OBJECT_WITH_QUERY_ACTION",
                }
            )
        )

        message = "张三这次考试怎么样"
        self.assertEqual(
            ScoreIntent.SCORE_QUERY,
            classify_score_intent(
                message,
                classifier=_fake_intent_classifier(0.95),
                llm_model=model,
            ),
        )
        prompt = model.invoke.call_args.args[0]
        self.assertIn("只输出JSON，不要解释", prompt)
        self.assertIn("<user_input>\n张三这次考试怎么样\n</user_input>", prompt)

    def test_llm_low_confidence_or_invalid_json_falls_back_to_general(self) -> None:
        low_confidence_model = mock.Mock()
        low_confidence_model.invoke.return_value = mock.Mock(
            content=json.dumps(
                {
                    "intent": "SCORE_QUERY",
                    "confidence": 0.59,
                    "reasonCode": "SCORE_RELATED_AMBIGUOUS",
                }
            )
        )
        invalid_model = mock.Mock()
        invalid_model.invoke.return_value = mock.Mock(content="not-json")

        self.assertEqual(
            ScoreIntent.GENERAL_QUERY,
            classify_score_intent_with_llm("考试", model=low_confidence_model),
        )
        self.assertEqual(
            ScoreIntent.GENERAL_QUERY,
            classify_score_intent_with_llm("考试", model=invalid_model),
        )

    def test_llm_classifies_every_non_query_intent_category(self) -> None:
        cases = (
            ("学校成绩制度是什么", ScoreIntent.SCORE_POLICY),
            ("老师怎么录入成绩", ScoreIntent.SCORE_OPERATION),
            ("我不是查成绩", ScoreIntent.GENERAL_QUERY),
            ("学校什么时候开学", ScoreIntent.GENERAL_QUERY),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                model = mock.Mock()
                model.invoke.return_value = mock.Mock(
                    content=json.dumps(
                        {
                            "intent": expected.value,
                            "confidence": 0.98,
                            "reasonCode": expected.value,
                        }
                    )
                )

                self.assertEqual(
                    expected,
                    classify_score_intent_with_llm(message, model=model),
                )
                model.invoke.assert_called_once()


class ScoreContextClientTest(unittest.TestCase):
    @mock.patch("rag.score_context_client.urllib.request.urlopen")
    def test_lookup_uses_signed_store_url_and_operation_contract(
        self,
        urlopen: mock.Mock,
    ) -> None:
        response = mock.MagicMock()
        response.read.return_value = json.dumps(
            {"data": {"hasClarifyingContext": True}}
        ).encode("utf-8")
        urlopen.return_value.__enter__.return_value = response
        env = {
            "WECHAT_SCORE_CONTEXT_API_URL": "http://backend.test/api/internal/wechat/score-query-contexts",
            "WECHAT_SCORE_CONTEXT_SERVICE_ID": "xinshi-rag",
            "WECHAT_SCORE_CONTEXT_SIGNING_SECRET": "test-signing-secret",
        }

        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("rag.score_context_client.time.time", return_value=1788172800),
            mock.patch(
                "rag.score_context_client.secrets.token_urlsafe",
                return_value="fixed-nonce",
            ),
        ):
            active = ScoreContextClient().has_active_clarification(
                appid="wx12345678", openid="openid"
            )

        self.assertTrue(active)
        request = urlopen.call_args.args[0]
        self.assertEqual(env["WECHAT_SCORE_CONTEXT_API_URL"], request.full_url)
        self.assertEqual(
            {
                "operation": "LOOKUP_CLARIFYING",
                "appid": "wx12345678",
                "openid": "openid",
            },
            json.loads(request.data.decode("utf-8")),
        )
        self.assertEqual("POST", request.method)
        self.assertEqual(
            b'{"operation":"LOOKUP_CLARIFYING","appid":"wx12345678","openid":"openid"}',
            request.data,
        )
        body_digest = hashlib.sha256(request.data).hexdigest()
        canonical = "\n".join(
            ("xinshi-rag", "1788172800", "fixed-nonce", body_digest)
        )
        expected_signature = hmac.new(
            b"test-signing-secret", canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        self.assertEqual(expected_signature, request.get_header("X-signature"))

    @mock.patch("rag.score_context_client.urllib.request.urlopen")
    def test_lookup_disabled_returns_false_without_request(
        self,
        urlopen: mock.Mock,
    ) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            active = ScoreContextClient().has_active_clarification(
                appid="wx12345678", openid="openid"
            )

        self.assertFalse(active)
        urlopen.assert_not_called()

    @mock.patch("rag.score_context_client.urllib.request.urlopen")
    def test_lookup_malformed_response_returns_false(self, urlopen: mock.Mock) -> None:
        response = mock.MagicMock()
        response.read.return_value = json.dumps(
            {"data": {"hasClarifyingContext": "true"}}
        ).encode("utf-8")
        urlopen.return_value.__enter__.return_value = response
        env = {
            "WECHAT_SCORE_CONTEXT_API_URL": "http://backend.test/api/internal/wechat/score-query-contexts",
            "WECHAT_SCORE_CONTEXT_SIGNING_SECRET": "test-signing-secret",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            active = ScoreContextClient().has_active_clarification(
                appid="wx12345678", openid="openid"
            )

        self.assertFalse(active)

    @mock.patch(
        "rag.score_context_client.urllib.request.urlopen",
        side_effect=urllib.error.URLError("backend unavailable"),
    )
    def test_lookup_network_failure_returns_false(self, _urlopen: mock.Mock) -> None:
        env = {
            "WECHAT_SCORE_CONTEXT_API_URL": "http://backend.test/api/internal/wechat/score-query-contexts",
            "WECHAT_SCORE_CONTEXT_SIGNING_SECRET": "test-signing-secret",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            client = ScoreContextClient()
            active = client.has_active_clarification(
                appid="wx12345678",
                openid="openid",
            )

        self.assertFalse(active)


class _StoredContext:
    def __init__(self, slots, missing_slot_codes) -> None:
        self.slots = slots
        self.missing_slot_codes = missing_slot_codes


class _ContextClient:
    def __init__(
        self,
        missing_slot_codes=(),
        *,
        active_clarification=False,
        clarification_error=None,
    ) -> None:
        self.missing_slot_codes = missing_slot_codes
        self.active_clarification = active_clarification
        self.clarification_error = clarification_error
        self.calls = []
        self.lookup_calls = []
        self.stored_slots = None

    def has_active_clarification(self, **kwargs):
        self.lookup_calls.append(kwargs)
        if self.clarification_error:
            raise self.clarification_error
        return self.active_clarification

    def store(self, **kwargs):
        self.calls.append(kwargs)
        incoming = kwargs["slots"]
        merged = _merge_slots(self.stored_slots, incoming)
        self.stored_slots = merged
        missing_slot_codes = self.missing_slot_codes or merged.missing_codes()
        self.active_clarification = bool(missing_slot_codes)
        return _StoredContext(merged, missing_slot_codes)


def _merge_slots(previous, incoming):
    if previous is None:
        return incoming
    return ScoreSlots(
        student_name=incoming.student_name or previous.student_name,
        year_value=incoming.year_value or previous.year_value,
        year_mode=incoming.year_mode or previous.year_mode,
        academic_year=incoming.academic_year or previous.academic_year,
        term_no=incoming.term_no or previous.term_no,
        exam_type_code=incoming.exam_type_code or previous.exam_type_code,
        subject_name=incoming.subject_name or previous.subject_name,
    )


class _ScoreWorkflowDriver:
    """Exercise score nodes in the same order used by the unified graph."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("slot_extractor", extract_score_slots)
        kwargs.setdefault("intent_classifier", _workflow_intent_classifier)
        self.workflow = ScoreQueryWorkflow(**kwargs)

    def invoke(
        self,
        *,
        message: str,
        channel: ChatChannel = ChatChannel.WECHAT,
        openid: str | None = "openid",
        msg_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        score_intent_enabled: bool = True,
    ) -> ChatGraphState:
        state: ChatGraphState = {
            "user_message": message,
            "history": history or [],
            "channel": channel,
            "channel_user_id": openid if channel is ChatChannel.WECHAT else None,
            "message_id": msg_id,
            "score_intent_enabled": score_intent_enabled,
        }
        state.update(self.workflow.load_score_context(state))
        state.update(self.workflow.classify_intent(state))
        route = self.workflow.route_after_intent(state)
        if route == "rag":
            return state
        if route == "intent_clarification":
            state.update(self.workflow.build_clarification(state))
            return state
        state.update(self.workflow.extract_slots(state))
        state.update(self.workflow.merge_or_persist_context(state))
        if self.workflow.route_after_context(state) == "clarify":
            state.update(self.workflow.build_clarification(state))
        else:
            state.update(self.workflow.build_h5_reply(state))
        return state


class ScoreQueryWorkflowTest(unittest.TestCase):
    def test_explicit_intent_does_not_call_llm_classifier(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.SCORE_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(
            message="查询张三2026年期末考试成绩",
            openid="openid",
            msg_id="explicit-1",
        )

        self.assertEqual(ChatResultType.SCORE_LINK, state["result_type"])
        classifier.assert_called_once_with("查询张三2026年期末考试成绩")
        self.assertEqual(1, len(client.lookup_calls))

    def test_ambiguous_message_uses_llm_result_before_routing(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.SCORE_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(message="考试", openid="openid", msg_id="ambiguous-1")

        classifier.assert_called_once_with("考试")
        self.assertEqual(ChatResultType.SCORE_CLARIFICATION, state["result_type"])
        self.assertIn("学生姓名", state["answer"])

    def test_general_question_ends_without_persisting_context(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.GENERAL_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(message="学校什么时候开学", openid="openid", msg_id="1")

        self.assertNotIn("result_type", state)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, state["intent"])
        classifier.assert_called_once_with("学校什么时候开学")
        self.assertEqual(1, len(client.lookup_calls))
        self.assertEqual([], client.calls)

    def test_slot_like_message_without_context_stays_general(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.GENERAL_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(message="张三，2026年期末", openid="openid", msg_id="1-1")

        self.assertNotIn("result_type", state)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, state["intent"])
        classifier.assert_called_once_with("张三，2026年期末")
        self.assertEqual([], client.calls)

    def test_active_clarification_merges_followup_and_builds_link(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.SCORE_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        first = graph.invoke(message="查成绩", openid="openid", msg_id="1-2")
        second = graph.invoke(
            message="查询张三2026年期末成绩",
            openid="openid",
            msg_id="1-3",
        )

        self.assertEqual(ChatResultType.SCORE_CLARIFICATION, first["result_type"])
        self.assertNotIn("example.test", first["answer"])
        self.assertEqual(ChatResultType.SCORE_LINK, second["result_type"])
        self.assertEqual("张三", second["score_slots"].student_name)
        self.assertEqual(2026, second["score_slots"].year_value)
        self.assertEqual("FINAL", second["score_slots"].exam_type_code)
        self.assertEqual((), second["missing_codes"])
        self.assertIn("http://example.test/h5/score", second["answer"])
        classifier.assert_called_once_with("查成绩")

    def test_active_clarification_uses_backend_merge(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.SCORE_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        first = graph.invoke(
            message="查询张三的成绩", openid="openid", msg_id="merge-1"
        )
        second = graph.invoke(
            message="2026年期末", openid="openid", msg_id="merge-2"
        )

        self.assertEqual(ChatResultType.SCORE_CLARIFICATION, first["result_type"])
        self.assertNotIn("example.test", first["answer"])
        self.assertEqual(2, len(client.calls))
        self.assertIsNone(client.calls[1]["slots"].student_name)
        self.assertEqual("merge-2", client.calls[1]["msg_id"])
        self.assertEqual(ChatResultType.SCORE_LINK, second["result_type"])
        self.assertEqual("张三", second["score_slots"].student_name)
        self.assertEqual((), second["missing_codes"])
        self.assertIn("http://example.test/h5/score", second["answer"])
        classifier.assert_called_once_with("查询张三的成绩")

    def test_clarification_lookup_failure_falls_back_to_general(self) -> None:
        client = _ContextClient(clarification_error=TimeoutError("backend timeout"))
        classifier = mock.Mock(return_value=ScoreIntent.GENERAL_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(message="张三，2026年期末", openid="openid", msg_id="1-4")

        self.assertNotIn("result_type", state)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, state["intent"])
        classifier.assert_called_once_with("张三，2026年期末")
        self.assertEqual([], client.calls)

    def test_active_clarification_does_not_override_explicit_rejection(self) -> None:
        client = _ContextClient(active_clarification=True)
        classifier = mock.Mock(return_value=ScoreIntent.GENERAL_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(
            message="不要查成绩，张三2026年期末",
            openid="openid",
            msg_id="1-5",
        )

        self.assertNotIn("result_type", state)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, state["intent"])
        classifier.assert_not_called()
        self.assertEqual([], client.calls)

    def test_active_clarification_does_not_override_policy_or_operation(self) -> None:
        client = _ContextClient(active_clarification=True)
        classifier = mock.Mock()
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        policy = graph.invoke(
            message="张三2026年期末成绩什么时候发布",
            openid="openid",
            msg_id="policy",
        )
        operation = graph.invoke(
            message="张三2026年期末成绩怎么导入",
            openid="openid",
            msg_id="operation",
        )

        self.assertNotIn("result_type", policy)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, policy["intent"])
        self.assertNotIn("result_type", operation)
        self.assertEqual(ScoreIntent.GENERAL_QUERY, operation["intent"])
        classifier.assert_not_called()
        self.assertEqual([], client.calls)

    def test_missing_slots_route_to_clarification(self) -> None:
        client = _ContextClient(("TIME_RANGE", "EXAM_TYPE"))
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
        )

        state = graph.invoke(message="查询张三的成绩", openid="openid", msg_id="2")

        self.assertEqual(ChatResultType.SCORE_CLARIFICATION, state["result_type"])
        self.assertIn("年份或学期", state["answer"])
        self.assertNotIn("example.test", state["answer"])

    def test_complete_slots_route_to_fixed_h5_link(self) -> None:
        client = _ContextClient()
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
        )

        state = graph.invoke(
            message="查询张三2026年期末考试成绩",
            openid="openid",
            msg_id="3",
        )

        self.assertEqual(ChatResultType.SCORE_LINK, state["result_type"])
        self.assertIn("http://example.test/h5/score", state["answer"])
        self.assertEqual("3", client.calls[0]["msg_id"])

    def test_web_history_merges_followup_without_wechat_context_calls(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock(return_value=ScoreIntent.SCORE_QUERY)
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )
        first = graph.invoke(
            message="查成绩",
            channel=ChatChannel.WEB,
            openid=None,
        )
        history = [
            {"role": "user", "content": "查成绩"},
            {"role": "assistant", "content": first["answer"]},
        ]

        second = graph.invoke(
            message="查询张三2026年期末成绩",
            channel=ChatChannel.WEB,
            openid=None,
            history=history,
        )

        self.assertEqual(ChatResultType.SCORE_LINK, second["result_type"])
        self.assertEqual("张三", second["score_slots"].student_name)
        self.assertEqual(2026, second["score_slots"].year_value)
        self.assertEqual([], client.lookup_calls)
        self.assertEqual([], client.calls)
        classifier.assert_called_once_with("查成绩")

    def test_disabled_score_intent_routes_to_rag(self) -> None:
        client = _ContextClient()
        classifier = mock.Mock()
        graph = _ScoreWorkflowDriver(
            enabled=True,
            appid="wx12345678",
            context_client=client,
            h5_url_provider=lambda: "http://example.test/h5/score?appid=wx12345678",
            intent_classifier=classifier,
        )

        state = graph.invoke(
            message="查询张三2026年期末考试成绩",
            channel=ChatChannel.STREAM,
            score_intent_enabled=False,
        )

        self.assertEqual(ScoreIntent.GENERAL_QUERY, state["intent"])
        self.assertNotIn("result_type", state)
        classifier.assert_not_called()
        self.assertEqual([], client.calls)


class UnifiedGraphStructureTest(unittest.TestCase):
    def test_chat_runtime_has_one_graph_compile_point(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        application_source = (project_root / "rag" / "application.py").read_text()
        score_source = (project_root / "rag" / "score_query_graph.py").read_text()

        self.assertEqual(1, application_source.count(".compile("))
        self.assertIn('name="xinshi-unified-chat"', application_source)
        self.assertNotIn("StateGraph", score_source)
        self.assertNotIn(".compile(", score_source)


class ScoreConfigurationCompatibilityTest(unittest.TestCase):
    def test_legacy_score_environment_names_remain_supported(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "WECHAT_SCORE_QUERY_ENABLED": "true",
                "WECHAT_SCORE_H5_URL": (
                    "http://example.test/h5/score?appid=wx12345678"
                ),
                "WECHAT_APPID": "wx12345678",
            },
            clear=True,
        ):
            workflow = ScoreQueryWorkflow.from_env()

        state: ChatGraphState = {
            "user_message": "查询张三2026年期末考试成绩",
            "history": [],
            "channel": ChatChannel.WEB,
            "score_intent_enabled": True,
        }
        state.update(workflow.load_score_context(state))
        with mock.patch(
            "rag.score_query_graph.classify_score_intent",
            return_value=ScoreIntent.SCORE_QUERY,
        ):
            state.update(workflow.classify_intent(state))
        self.assertEqual("score_query", workflow.route_after_intent(state))
        state["score_slots"] = extract_score_slots(state["user_message"])
        state.update(workflow.merge_or_persist_context(state))
        state.update(workflow.build_h5_reply(state))
        self.assertIn("/h5/score?appid=wx12345678", state["answer"])

    def test_new_disabled_flag_overrides_legacy_enabled_flag(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCORE_QUERY_ENABLED": "false",
                "WECHAT_SCORE_QUERY_ENABLED": "true",
            },
            clear=True,
        ):
            workflow = ScoreQueryWorkflow.from_env()

        state: ChatGraphState = {
            "user_message": "查询张三2026年期末考试成绩",
            "history": [],
            "channel": ChatChannel.WEB,
            "score_intent_enabled": True,
        }
        state.update(workflow.load_score_context(state))
        with mock.patch(
            "rag.score_query_graph.classify_score_intent",
            return_value=ScoreIntent.SCORE_QUERY,
        ):
            state.update(workflow.classify_intent(state))
        self.assertEqual("rag", workflow.route_after_intent(state))


class WeChatCallbackSecurityTest(unittest.TestCase):
    def test_wechat_config_has_no_usable_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = WeChatConfig.from_env()
            with self.assertRaises(WeChatCryptoError):
                config.validate()

    @mock.patch("rag.wechat_service._dispatch_chat_request")
    @mock.patch("rag.wechat_service.decrypt_text_payload")
    @mock.patch("rag.wechat_service.verify_msg_signature", return_value=True)
    def test_reply_target_only_uses_decrypted_from_user(
        self,
        _verify: mock.Mock,
        decrypt: mock.Mock,
        handler: mock.Mock,
    ) -> None:
        decrypt.return_value = (
            "<xml><FromUserName><![CDATA[openid-from-message]]></FromUserName>"
            "<MsgType><![CDATA[text]]></MsgType><Content><![CDATA[查成绩]]></Content>"
            "<MsgId>123</MsgId></xml>"
        )

        result = build_encrypted_chat_reply(
            {"Encrypt": "ciphertext"},
            msg_signature="signature",
            timestamp="1",
            nonce="2",
            signature_present=False,
        )

        self.assertIsNone(result)
        self.assertEqual(
            "openid-from-message",
            handler.call_args.kwargs["wechat_openid"],
        )
        self.assertEqual("123", handler.call_args.kwargs["wechat_msg_id"])

    @mock.patch("rag.wechat_service._dispatch_chat_request")
    @mock.patch("rag.wechat_service.decrypt_text_payload")
    @mock.patch("rag.wechat_service.verify_msg_signature", return_value=True)
    def test_missing_from_user_is_ignored(
        self,
        _verify: mock.Mock,
        decrypt: mock.Mock,
        handler: mock.Mock,
    ) -> None:
        decrypt.return_value = (
            "<xml><MsgType><![CDATA[text]]></MsgType>"
            "<Content><![CDATA[查询张三2026年期末成绩]]></Content></xml>"
        )

        result = build_encrypted_chat_reply(
            {"Encrypt": "ciphertext"},
            msg_signature="signature",
            timestamp="1",
            nonce="2",
            signature_present=False,
        )

        self.assertIsNone(result)
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
