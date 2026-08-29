from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from arg.xinshi.config import COLLECTION_NAME

log = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency path
    from pymilvus import connections as _connections
    from pymilvus import utility as _utility
except Exception:  # pragma: no cover - optional dependency path
    _connections = None  # type: ignore[assignment]
    _utility = None  # type: ignore[assignment]


@dataclass
class _CollectionRegistry:
    names: set[str]


_REGISTRY = _CollectionRegistry(names=set())


def _connect(host: str, port: str) -> None:
    if _connections is not None:
        _connections.connect(alias="default", host=host, port=port)
        return
    log.info("pymilvus is unavailable, using local no-op registry")


def _has_collection(name: str) -> bool:
    if _utility is not None:
        return bool(_utility.has_collection(name))
    return name in _REGISTRY.names


def _drop_collection(name: str) -> None:
    if _utility is not None:
        _utility.drop_collection(name)
        return
    _REGISTRY.names.discard(name)


for i in range(10):
    try:
        _connect(host="localhost", port="19530")
        print("Connected!")
        break
    except Exception as e:
        print("Retrying...", e)
        time.sleep(5)


def drop_collection(name):
    collection_name = name
    if _has_collection(collection_name):
        _drop_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


drop_collection(COLLECTION_NAME)
