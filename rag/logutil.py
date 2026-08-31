"""统一日志配置。环境变量：XINSHI_LOG_LEVEL（默认 INFO），如 DEBUG / WARNING。"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("XINSHI_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = (
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(h)

    # 降噪：第三方库默认 WARNING
    for name in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def preview_text(text: str, max_len: int = 120) -> str:
    """日志里预览文本，避免过长；不记录完整用户输入到 DEBUG 以外时可配合使用。"""
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."
