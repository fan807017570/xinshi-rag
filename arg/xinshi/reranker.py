from __future__ import annotations

import heapq
import hashlib
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Sequence

from arg.xinshi.config import *

log = logging.getLogger(__name__)

_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+")

try:  # pragma: no cover - optional dependency path
    from sentence_transformers import CrossEncoder as _CrossEncoder
except Exception:  # pragma: no cover - optional dependency path
    _CrossEncoder = None  # type: ignore[assignment]


class _FallbackCrossEncoder:
    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, pair_batch, batch_size=64, show_progress_bar=False):
        scores = []
        for query, content in pair_batch:
            scores.append(_score_pair(query, content))
        return scores


def _text_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _TERM_RE.finditer(text):
        token = match.group(0).lower()
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
    return terms


def _score_pair(query: str, content: str) -> float:
    query = (query or "").strip()
    content = (content or "").strip()
    if not query or not content:
        return 0.0

    q_terms = set(_text_terms(query))
    c_terms = set(_text_terms(content))
    if not q_terms or not c_terms:
        return 0.0

    overlap = sum(max(1, len(term)) for term in q_terms & c_terms)
    total = sum(max(1, len(term)) for term in q_terms) or 1
    return min(1.0, overlap / total)


class BGEReranker:
    def __init__(self):
        self.batch_size = RERANK_BATCH_SIZE
        self.max_length = RERANK_MAX_LENGTH
        self.backend = RERANK_BACKEND
        self.candidate_limit = RERANK_CANDIDATE_LIMIT
        self.query_char_limit = RERANK_QUERY_CHAR_LIMIT
        self.doc_char_limit = RERANK_DOC_CHAR_LIMIT
        self.cache_size = RERANK_SCORE_CACHE_SIZE
        self._score_cache: OrderedDict[str, float] = OrderedDict()
        self._fallback_mode = False
        try:
            if _CrossEncoder is None:
                raise RuntimeError("sentence_transformers is unavailable")
            self.model = _CrossEncoder(
                RERANK_MODEL,
                max_length=self.max_length,
                backend=self.backend,
            )
        except Exception:
            log.warning(
                "reranker backend=%s unavailable, using local fallback",
                self.backend,
            )
            self.backend = "local"
            self._fallback_mode = True
            self.model = _FallbackCrossEncoder()
        log.info(
            (
                "reranker ready model=%s backend=%s batch_size=%d max_length=%d "
                "candidate_limit=%d query_char_limit=%d doc_char_limit=%d cache_size=%d"
            ),
            RERANK_MODEL,
            self.backend,
            self.batch_size,
            self.max_length,
            self.candidate_limit,
            self.query_char_limit,
            self.doc_char_limit,
            self.cache_size,
        )

    def _make_cache_key(self, query: str, content: str) -> str:
        material = f"{query}\0{content}".encode("utf-8", errors="ignore")
        return hashlib.sha1(material).hexdigest()

    def _cache_get(self, key: str):
        if self.cache_size <= 0:
            return None
        value = self._score_cache.get(key)
        if value is not None:
            self._score_cache.move_to_end(key)
        return value

    def _cache_set(self, key: str, value: float) -> None:
        if self.cache_size <= 0:
            return
        self._score_cache[key] = value
        self._score_cache.move_to_end(key)
        while len(self._score_cache) > self.cache_size:
            self._score_cache.popitem(last=False)

    def rerank(self, query, docs, top_k):
        if not docs or top_k <= 0:
            return []
        if len(docs) == 1:
            return docs[:1]
        if top_k >= len(docs):
            return docs

        t0 = time.perf_counter()
        top_k = min(top_k, len(docs))

        docs_for_rerank = docs[:self.candidate_limit]
        if top_k >= len(docs_for_rerank):
            return docs[:top_k]

        safe_query = (query or "")[:self.query_char_limit]
        scores = [None] * len(docs_for_rerank)
        uncached_pairs = []
        uncached_meta = []

        for idx, doc in enumerate(docs_for_rerank):
            safe_content = (doc.page_content or "")[:self.doc_char_limit]
            key = self._make_cache_key(safe_query, safe_content)
            cached_score = self._cache_get(key)
            if cached_score is not None:
                scores[idx] = cached_score
                continue
            uncached_pairs.append((safe_query, safe_content))
            uncached_meta.append((idx, key))

        for i in range(0, len(uncached_pairs), self.batch_size):
            pair_batch = uncached_pairs[i:i + self.batch_size]
            meta_batch = uncached_meta[i:i + self.batch_size]
            batch_scores = self.model.predict(
                pair_batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            if hasattr(batch_scores, "tolist"):
                numeric_scores = batch_scores.tolist()
            else:
                numeric_scores = list(batch_scores)

            for (idx, key), score in zip(meta_batch, numeric_scores):
                score = float(score)
                scores[idx] = score
                self._cache_set(key, score)

        all_scores = [s if s is not None else float("-inf") for s in scores]

        ranked = heapq.nlargest(
            top_k,
            zip(docs_for_rerank, all_scores),
            key=lambda x: x[1]
        )
        log.info(
            (
                "rerank done docs=%d rerank_docs=%d uncached=%d top_k=%d "
                "elapsed=%.3fs batch_size=%d max_length=%d"
            ),
            len(docs),
            len(docs_for_rerank),
            len(uncached_pairs),
            top_k,
            time.perf_counter() - t0,
            self.batch_size,
            self.max_length,
        )
        return [doc for doc, _ in ranked[:top_k]]
