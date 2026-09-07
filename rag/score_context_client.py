from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from rag.score_slot_extractor import ScoreSlots

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredScoreContext:
    context_status: str
    missing_slot_codes: tuple[str, ...]
    slots: ScoreSlots


class ScoreContextClient:
    def __init__(self) -> None:
        self._url = os.environ.get("WECHAT_SCORE_CONTEXT_API_URL", "").strip()
        self._service_id = os.environ.get(
            "WECHAT_SCORE_CONTEXT_SERVICE_ID", "xinshi-rag"
        ).strip()
        self._secret = os.environ.get("WECHAT_SCORE_CONTEXT_SIGNING_SECRET", "").strip()
        self._timeout = max(
            1,
            int(os.environ.get("WECHAT_SCORE_CONTEXT_TIMEOUT_SECONDS", "3")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._service_id and self._secret)

    def has_active_clarification(self, *, appid: str, openid: str) -> bool:
        """Returns whether the backend has an active slot clarification.

        The lookup is deliberately fail-closed. An unavailable or malformed
        backend response must leave the message on the ordinary RAG path.
        """
        if not self.enabled:
            return False
        body = _json_body(
            {
                "operation": "LOOKUP_CLARIFYING",
                "appid": appid,
                "openid": openid,
            }
        )
        try:
            payload = self._signed_post(self._url, body)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            log.exception(
                "score clarification lookup failed appid=%s identity=%s",
                appid,
                _digest(openid),
            )
            return False
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return False
        return data.get("hasClarifyingContext") is True

    def store(
        self,
        *,
        appid: str,
        openid: str,
        msg_id: str | None,
        slots: ScoreSlots,
    ) -> StoredScoreContext | None:
        if not self.enabled:
            return None
        body = _json_body(
            {
                "appid": appid,
                "openid": openid,
                "msgId": msg_id,
                "intent": "SCORE_QUERY",
                "contextStatus": "PENDING" if not slots.missing_codes() else "CLARIFYING",
                "missingSlotCodes": list(slots.missing_codes()),
                "slots": slots.to_payload(),
            }
        )
        try:
            payload = self._signed_post(self._url, body)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            log.exception(
                "score context store failed appid=%s identity=%s",
                appid,
                _digest(openid),
            )
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return StoredScoreContext(
                context_status="PENDING" if not slots.missing_codes() else "CLARIFYING",
                missing_slot_codes=slots.missing_codes(),
                slots=slots,
            )
        merged_slots = _slots_from_payload(data.get("slots"), slots)
        missing_codes = tuple(
            str(code)
            for code in data.get("missingSlotCodes") or merged_slots.missing_codes()
        )
        return StoredScoreContext(
            context_status=str(data.get("contextStatus") or "CLARIFYING"),
            missing_slot_codes=missing_codes,
            slots=merged_slots,
        )

    def _signed_post(self, url: str, body: bytes) -> object:
        timestamp = str(int(time.time()))
        nonce = secrets.token_urlsafe(18)
        body_digest = hashlib.sha256(body).hexdigest()
        canonical = "\n".join((self._service_id, timestamp, nonce, body_digest))
        signature = hmac.new(
            self._secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Service-Id": self._service_id,
                "X-Timestamp": timestamp,
                "X-Nonce": nonce,
                "X-Signature": signature,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def _json_body(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _slots_from_payload(payload: object, fallback: ScoreSlots) -> ScoreSlots:
    if not isinstance(payload, dict):
        return fallback
    year_value = payload.get("yearValue")
    return ScoreSlots(
        student_name=(
            _optional_string(payload.get("studentName")) or fallback.student_name
        ),
        year_value=int(year_value) if year_value is not None else fallback.year_value,
        year_mode=_optional_string(payload.get("yearMode")) or fallback.year_mode,
        academic_year=(
            _optional_string(payload.get("academicYear")) or fallback.academic_year
        ),
        term_no=(
            int(payload["termNo"])
            if payload.get("termNo") is not None
            else fallback.term_no
        ),
        exam_type_code=(
            _optional_string(payload.get("examTypeCode"))
            or fallback.exam_type_code
        ),
        subject_name=(
            _optional_string(payload.get("subjectName")) or fallback.subject_name
        ),
    )


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
