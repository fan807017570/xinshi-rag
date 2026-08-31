"""Strict DeepSeek LLM configuration and request handling."""

from __future__ import annotations

import os
from typing import Any, Iterator

try:
    from langchain_deepseek import ChatDeepSeek
except Exception:  # pragma: no cover - dependency validation handles this
    ChatDeepSeek = None  # type: ignore[assignment]


class LLMConfigurationError(RuntimeError):
    """Raised when required DeepSeek configuration is missing or invalid."""


class LLMRequestError(RuntimeError):
    """Raised when a configured DeepSeek request cannot be completed."""


class StrictChatModel:
    """Expose the small chat-model interface used by the RAG application."""

    def __init__(self, client: Any, model_name: str):
        self._client = client
        self.model_name = model_name

    def invoke(self, prompt: str):
        """Invoke DeepSeek or raise a sanitized request error."""
        try:
            return self._client.invoke(prompt)
        except Exception as exc:
            raise _request_error(exc) from exc

    def stream(self, prompt: str) -> Iterator[Any]:
        """Stream DeepSeek output or raise a sanitized request error."""
        try:
            yield from self._client.stream(prompt)
        except Exception as exc:
            raise _request_error(exc) from exc


def _env_int(name: str, default: int, minimum: int) -> int:
    """Read a validated integer environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} 必须是整数，当前值无效"
        ) from exc
    if value < minimum:
        raise LLMConfigurationError(
            f"{name} 必须大于等于 {minimum}，当前值为 {value}"
        )
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a validated floating-point environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} 必须是数字，当前值无效"
        ) from exc
    if not minimum <= value <= maximum:
        raise LLMConfigurationError(
            f"{name} 必须在 {minimum} 到 {maximum} 之间，当前值为 {value}"
        )
    return value


def _create_chat_model(
    *,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> StrictChatModel:
    """Create DeepSeek without any local fallback behavior."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "未配置 DEEPSEEK_API_KEY；请在 rag/.env 或运行环境中设置有效 Key"
        )
    if ChatDeepSeek is None:
        raise LLMConfigurationError(
            "langchain-deepseek 依赖不可用，请重新安装项目依赖"
        )

    normalized_model_name = model_name.strip()
    if not normalized_model_name:
        raise LLMConfigurationError("DeepSeek 模型名称不能为空")

    try:
        client = ChatDeepSeek(
            model=normalized_model_name,
            api_key=api_key,
            streaming=True,
            stream_usage=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise LLMConfigurationError(
            "DeepSeek 客户端初始化失败，请检查模型名称和 LLM 参数"
        ) from exc
    return StrictChatModel(client, normalized_model_name)


def get_llm() -> StrictChatModel:
    """Return the strictly configured answer-generation model."""
    return _create_chat_model(
        model_name=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_tokens=_env_int("DEEPSEEK_MAX_TOKENS", 1024, 128),
        temperature=_env_float(
            "DEEPSEEK_TEMPERATURE",
            0.0,
            minimum=0.0,
            maximum=2.0,
        ),
    )


def get_little_llm() -> StrictChatModel:
    """Return the strictly configured query-rewrite model."""
    return _create_chat_model(
        model_name=os.environ.get(
            "DEEPSEEK_MODEL_LITTLE",
            os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        ),
        max_tokens=_env_int("DEEPSEEK_MAX_TOKENS_LITTLE", 128, 32),
        temperature=_env_float(
            "DEEPSEEK_TEMPERATURE_LITTLE",
            0.2,
            minimum=0.0,
            maximum=2.0,
        ),
    )


def get_llm_runtime_status() -> dict[str, str]:
    """Validate local LLM settings and return non-sensitive backend details."""
    answer_model = get_llm()
    rewrite_model = get_little_llm()
    return {
        "llm_backend": "deepseek",
        "llm_model": answer_model.model_name,
        "llm_rewrite_model": rewrite_model.model_name,
    }


def _request_error(exc: Exception) -> LLMRequestError:
    """Build an error message that never contains credentials."""
    return LLMRequestError(
        "DeepSeek 请求失败"
        f"（{type(exc).__name__}）；请检查 API Key、模型名称、账户状态和网络配置"
    )
