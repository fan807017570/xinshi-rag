"""Load RAG runtime settings from an external INI file.

Settings are loaded once when the application process starts. Environment
variables remain supported as compatibility overrides, with higher priority
than the configuration file.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "application.ini"
CONFIG_FILE = Path(
    os.environ.get("XINSHI_CONFIG_FILE", str(DEFAULT_CONFIG_FILE))
).expanduser()


def _load_config(config_file: Path) -> configparser.ConfigParser:
    """Read and validate the external configuration file.

    Args:
        config_file: INI file to read.

    Returns:
        Parsed configuration.

    Raises:
        FileNotFoundError: If the configured file does not exist.
        ValueError: If the file is not valid INI syntax.
    """
    if not config_file.is_file():
        raise FileNotFoundError(
            f"xinshi-rag configuration file does not exist: {config_file}"
        )

    parser = configparser.ConfigParser(interpolation=None)
    try:
        with config_file.open("r", encoding="utf-8") as config_stream:
            parser.read_file(config_stream)
    except configparser.Error as exc:
        raise ValueError(
            f"invalid xinshi-rag configuration file {config_file}: {exc}"
        ) from exc
    return parser


_CONFIG = _load_config(CONFIG_FILE)
CONFIG_PROFILE = os.environ.get(
    "XINSHI_CONFIG_PROFILE",
    _CONFIG.get("runtime", "profile", fallback="native"),
).strip()


def _get_value(section: str, option: str, environment_name: str) -> str:
    """Return an environment override or a profile-aware INI value."""
    environment_value = os.environ.get(environment_name)
    if environment_value is not None:
        return environment_value.strip()

    profile_section = f"{section}:{CONFIG_PROFILE}"
    if _CONFIG.has_option(profile_section, option):
        return _CONFIG.get(profile_section, option).strip()
    if _CONFIG.has_option(section, option):
        return _CONFIG.get(section, option).strip()

    raise KeyError(
        f"missing setting [{section}] {option} in {CONFIG_FILE} "
        f"for profile {CONFIG_PROFILE!r}"
    )


def _get_int(
    section: str,
    option: str,
    environment_name: str,
    minimum: int,
) -> int:
    """Return an integer setting constrained to a minimum value."""
    raw_value = _get_value(section, option, environment_name)
    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"setting [{section}] {option} must be an integer, got {raw_value!r}"
        ) from exc
    return max(minimum, parsed_value)


def _get_bool(section: str, option: str, environment_name: str) -> bool:
    """Return a strict boolean setting."""
    raw_value = _get_value(section, option, environment_name).lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"setting [{section}] {option} must be a boolean, got {raw_value!r}"
    )


def _get_project_path(
    section: str,
    option: str,
    environment_name: str,
) -> str:
    """Return an absolute path, resolving relative values from project root."""
    configured_path = Path(
        _get_value(section, option, environment_name)
    ).expanduser()
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return str(configured_path.resolve())


USE_MILVUS = _get_bool("milvus", "enabled", "XINSHI_USE_MILVUS")
MILVUS_HOST = _get_value("milvus", "host", "MILVUS_HOST")
MILVUS_PORT = _get_value("milvus", "port", "MILVUS_PORT")
COLLECTION_NAME = _get_value("milvus", "collection", "MILVUS_COLLECTION")

EMBEDDING_MODEL = _get_project_path(
    "models", "embedding_model", "EMBEDDING_MODEL"
)
EMBEDDING_MODEL_CACHE_DIR = _get_project_path(
    "models", "embedding_model_cache_dir", "EMBEDDING_MODEL_CACHE_DIR"
)
RERANK_MODEL = _get_project_path("models", "rerank_model", "RERANK_MODEL")
RERANK_MODEL_CACHE_DIR = _get_project_path(
    "models", "rerank_model_cache_dir", "RERANK_MODEL_CACHE_DIR"
)

RERANK_BATCH_SIZE = _get_int(
    "reranker", "batch_size", "RERANK_BATCH_SIZE", minimum=1
)
RERANK_MAX_LENGTH = _get_int(
    "reranker", "max_length", "RERANK_MAX_LENGTH", minimum=64
)
RERANK_BACKEND = _get_value("reranker", "backend", "RERANK_BACKEND")
RERANK_CANDIDATE_LIMIT = _get_int(
    "reranker", "candidate_limit", "RERANK_CANDIDATE_LIMIT", minimum=1
)
RERANK_QUERY_CHAR_LIMIT = _get_int(
    "reranker", "query_char_limit", "RERANK_QUERY_CHAR_LIMIT", minimum=16
)
RERANK_DOC_CHAR_LIMIT = _get_int(
    "reranker", "doc_char_limit", "RERANK_DOC_CHAR_LIMIT", minimum=64
)
RERANK_SCORE_CACHE_SIZE = _get_int(
    "reranker", "score_cache_size", "RERANK_SCORE_CACHE_SIZE", minimum=0
)

TOP_K_RETRIEVE = _get_int(
    "retrieval", "top_k_retrieve", "TOP_K_RETRIEVE", minimum=1
)
TOP_K_RERANK = _get_int(
    "retrieval", "top_k_rerank", "TOP_K_RERANK", minimum=1
)
ROLE_FILTER_MULTIPLIER = _get_int(
    "retrieval",
    "role_filter_multiplier",
    "ROLE_FILTER_MULTIPLIER",
    minimum=1,
)
_MAX_RETRIEVAL_CANDIDATES = TOP_K_RETRIEVE * ROLE_FILTER_MULTIPLIER
if RERANK_CANDIDATE_LIMIT < _MAX_RETRIEVAL_CANDIDATES:
    raise ValueError(
        "setting [reranker] candidate_limit must be at least "
        "[retrieval] top_k_retrieve * role_filter_multiplier "
        f"({_MAX_RETRIEVAL_CANDIDATES}), got {RERANK_CANDIDATE_LIMIT}"
    )
MAX_HISTORY_MESSAGES = _get_int(
    "conversation",
    "max_history_messages",
    "MAX_HISTORY_MESSAGES",
    minimum=1,
)


__all__ = [
    "COLLECTION_NAME",
    "CONFIG_FILE",
    "CONFIG_PROFILE",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_CACHE_DIR",
    "MAX_HISTORY_MESSAGES",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "RERANK_BACKEND",
    "RERANK_BATCH_SIZE",
    "RERANK_CANDIDATE_LIMIT",
    "RERANK_DOC_CHAR_LIMIT",
    "RERANK_MAX_LENGTH",
    "RERANK_MODEL",
    "RERANK_MODEL_CACHE_DIR",
    "RERANK_QUERY_CHAR_LIMIT",
    "RERANK_SCORE_CACHE_SIZE",
    "ROLE_FILTER_MULTIPLIER",
    "TOP_K_RERANK",
    "TOP_K_RETRIEVE",
    "USE_MILVUS",
]
