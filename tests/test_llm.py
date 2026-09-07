from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from rag import llm


class _WorkingClient:
    def invoke(self, _prompt: str):
        return SimpleNamespace(content="ok")

    def stream(self, _prompt: str):
        yield SimpleNamespace(content="ok")


class _FailingClient:
    def invoke(self, _prompt: str):
        raise RuntimeError("sensitive upstream detail")

    def stream(self, _prompt: str):
        raise RuntimeError("sensitive upstream detail")
        yield


class LlmConfigurationTest(unittest.TestCase):
    def test_missing_api_key_raises_configuration_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                llm.LLMConfigurationError,
                "DEEPSEEK_API_KEY",
            ):
                llm.get_llm()

    def test_invalid_numeric_setting_raises_configuration_error(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_MAX_TOKENS": "invalid",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                llm.LLMConfigurationError,
                "DEEPSEEK_MAX_TOKENS",
            ):
                llm.get_llm()

    def test_client_initialization_failure_is_not_downgraded(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "test-key"},
                clear=True,
            ),
            mock.patch.object(
                llm,
                "ChatDeepSeek",
                side_effect=ValueError("invalid configuration"),
            ),
        ):
            with self.assertRaisesRegex(
                llm.LLMConfigurationError,
                "初始化失败",
            ):
                llm.get_llm()

    def test_request_failure_returns_sanitized_error(self) -> None:
        model = llm.StrictChatModel(_FailingClient(), "model")

        with self.assertRaises(llm.LLMRequestError) as raised:
            model.invoke("prompt")

        self.assertNotIn("sensitive upstream detail", str(raised.exception))
        self.assertIn("DeepSeek 请求失败", str(raised.exception))

    def test_stream_failure_returns_sanitized_error(self) -> None:
        model = llm.StrictChatModel(_FailingClient(), "model")

        with self.assertRaises(llm.LLMRequestError) as raised:
            list(model.stream("prompt"))

        self.assertNotIn("sensitive upstream detail", str(raised.exception))
        self.assertIn("DeepSeek 请求失败", str(raised.exception))

    def test_configured_model_is_returned(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "test-key"},
                clear=True,
            ),
            mock.patch.object(
                llm,
                "ChatDeepSeek",
                return_value=_WorkingClient(),
            ),
        ):
            model = llm.get_llm()

        self.assertEqual(model.invoke("prompt").content, "ok")
        self.assertEqual(model.model_name, "deepseek-v4-flash")

    def test_intent_model_uses_zero_temperature_and_configured_timeout(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "WECHAT_SCORE_INTENT_LLM_TIMEOUT_MS": "1700",
                },
                clear=True,
            ),
            mock.patch.object(
                llm,
                "ChatDeepSeek",
                return_value=_WorkingClient(),
            ) as chat_deepseek,
        ):
            model = llm.get_intent_llm()

        self.assertEqual(model.invoke("prompt").content, "ok")
        self.assertEqual(
            {
                "model": "deepseek-v4-flash",
                "api_key": "test-key",
                "streaming": True,
                "stream_usage": True,
                "max_tokens": 128,
                "temperature": 0.0,
                "timeout": 1.7,
            },
            chat_deepseek.call_args.kwargs,
        )

    def test_shared_intent_timeout_overrides_legacy_wechat_name(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "SCORE_INTENT_LLM_TIMEOUT_MS": "2100",
                    "WECHAT_SCORE_INTENT_LLM_TIMEOUT_MS": "1700",
                },
                clear=True,
            ),
            mock.patch.object(
                llm,
                "ChatDeepSeek",
                return_value=_WorkingClient(),
            ) as chat_deepseek,
        ):
            llm.get_intent_llm()

        self.assertEqual(2.1, chat_deepseek.call_args.kwargs["timeout"])

    def test_score_slot_model_uses_dedicated_timeout_and_zero_temperature(
        self,
    ) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "DEEPSEEK_MODEL_SCORE_SLOT": "score-slot-model",
                    "SCORE_SLOT_LLM_TIMEOUT_MS": "2300",
                },
                clear=True,
            ),
            mock.patch.object(
                llm,
                "ChatDeepSeek",
                return_value=_WorkingClient(),
            ) as chat_deepseek,
        ):
            model = llm.get_score_slot_llm()

        self.assertEqual("ok", model.invoke("prompt").content)
        self.assertEqual(
            {
                "model": "score-slot-model",
                "api_key": "test-key",
                "streaming": True,
                "stream_usage": True,
                "max_tokens": 256,
                "temperature": 0.0,
                "timeout": 2.3,
            },
            chat_deepseek.call_args.kwargs,
        )


if __name__ == "__main__":
    unittest.main()
