from __future__ import annotations

import logging
import os
import re
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from arg.xinshi.application import answer_question, stream_answer_question
from arg.xinshi.config import MAX_HISTORY_MESSAGES
from arg.xinshi.logutil import preview_text
from arg.xinshi.schemas import ChatRequest, ChatResponse

log = logging.getLogger(__name__)

_APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Shanghai")
_WECHAT_APPID = os.environ.get("WECHAT_APPID", "").strip()
_WECHAT_APPSECRET = os.environ.get("WECHAT_APPSECRET", "").strip()
_WECHAT_ACCESS_TOKEN = os.environ.get("WECHAT_ACCESS_TOKEN", "").strip()
_WECHAT_TOKEN_EXPIRE_SKEW_SECONDS = 120
_WECHAT_CUSTOM_REPLY_WORKERS = max(1, int(os.environ.get("WECHAT_CUSTOM_REPLY_WORKERS", "6")))
_WECHAT_CUSTOM_REPLY_QUEUE_SIZE = max(0, int(os.environ.get("WECHAT_CUSTOM_REPLY_QUEUE_SIZE", "150")))
_WECHAT_CUSTOM_REPLY_TIMEOUT = max(1, int(os.environ.get("WECHAT_CUSTOM_REPLY_TIMEOUT", "10")))
_WECHAT_CUSTOM_REPLY_MAX_BYTES = max(256, int(os.environ.get("WECHAT_CUSTOM_REPLY_MAX_BYTES", "1800")))
_WECHAT_BUSY_MESSAGE = "当前咨询人数较多，请稍后再试"
_WECHAT_TOKEN_API = "https://api.weixin.qq.com/cgi-bin/token"
_WECHAT_CUSTOM_SEND_API = "https://api.weixin.qq.com/cgi-bin/message/custom/send"
_wechat_reply_executor = ThreadPoolExecutor(
    max_workers=_WECHAT_CUSTOM_REPLY_WORKERS,
    thread_name_prefix="wechat-reply",
)
_wechat_reply_slots = threading.BoundedSemaphore(
    _WECHAT_CUSTOM_REPLY_WORKERS + _WECHAT_CUSTOM_REPLY_QUEUE_SIZE
)
_wechat_token_lock = threading.Lock()
_wechat_cached_access_token = ""
_wechat_cached_access_token_expires_at = 0.0
_MACHINE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~+=:/-]+$")
_HEX_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
_BASE64ISH_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/]{12,}={0,2}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_DATE_QUERY_RE = re.compile(
    r"(今天|今日|现在|当前).*(几号|日期|星期几|周几|礼拜几)"
    r"|今天[是]?几月几号"
    r"|今天[是]?什么日期"
)
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def is_non_natural_language_message(message: str) -> bool:
    text = message.strip()
    if not text:
        return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return False
    if re.search(r"\s", text):
        return False
    if any(ch in "?!！？" for ch in text):
        return False

    if _UUID_RE.fullmatch(text) or text.isdigit() or _HEX_TOKEN_RE.fullmatch(text):
        return True
    if _BASE64ISH_TOKEN_RE.fullmatch(text):
        return True
    if _MACHINE_TOKEN_RE.fullmatch(text):
        has_digit = any(ch.isdigit() for ch in text)
        has_separator = any(ch in "._~+=:/-" for ch in text)
        return len(text) >= 16 or (has_digit and len(text) >= 6) or has_separator
    return not any(ch.isalpha() for ch in text)


def _direct_chat_response(message: str) -> ChatResponse:
    return ChatResponse(
        answer=message.strip(),
        sources=[],
        rewritten_query=None,
        standalone_query=None,
    )


def _now() -> datetime:
    try:
        return datetime.now(ZoneInfo(_APP_TIMEZONE))
    except ZoneInfoNotFoundError:
        log.warning("APP_TIMEZONE=%s is invalid, fallback to local timezone", _APP_TIMEZONE)
        return datetime.now()


def _date_answer(message: str) -> str | None:
    text = re.sub(r"\s+", "", message.strip())
    if not _DATE_QUERY_RE.search(text):
        return None

    now = _now()
    weekday = _WEEKDAYS[now.weekday()]
    return f"今天是{now.year}年{now.month}月{now.day}日，{weekday}。"


def _date_chat_response(message: str) -> ChatResponse | None:
    answer = _date_answer(message)
    if answer is None:
        return None
    log.info("chat short-circuited date question answer=%r", answer)
    return ChatResponse(
        answer=answer,
        sources=[],
        rewritten_query=None,
        standalone_query=None,
    )


def _history_for(req: ChatRequest) -> list[dict[str, str]]:
    return [m.model_dump() for m in req.history[-MAX_HISTORY_MESSAGES:]]


def _chat_response(req: ChatRequest) -> ChatResponse:
    date_resp = _date_chat_response(req.message)
    if date_resp is not None:
        return date_resp

    if is_non_natural_language_message(req.message):
        log.info(
            "chat short-circuited non-natural message_len=%d preview=%r",
            len(req.message),
            preview_text(req.message, 80),
        )
        return _direct_chat_response(req.message)

    hist = _history_for(req)
    log.info(
        "chat request use_rewrite=%s history=%d message_len=%d preview=%r",
        req.use_rewrite,
        len(hist),
        len(req.message),
        preview_text(req.message, 80),
    )
    out = answer_question(
        req.message.strip(),
        history=hist,
        use_rewrite=req.use_rewrite,
    )
    return ChatResponse(
        answer=out["answer"],
        sources=out["sources"],
        rewritten_query=out.get("rewritten_query"),
        standalone_query=out.get("standalone_query"),
    )


def _empty_chat_response() -> ChatResponse:
    return ChatResponse(answer="", sources=[], rewritten_query=None, standalone_query=None)


def _get_wechat_access_token(*, force_refresh: bool = False) -> str:
    global _wechat_cached_access_token, _wechat_cached_access_token_expires_at

    if _WECHAT_ACCESS_TOKEN and not force_refresh:
        return _WECHAT_ACCESS_TOKEN

    now = time.time()
    with _wechat_token_lock:
        if (
            not force_refresh
            and _wechat_cached_access_token
            and now < _wechat_cached_access_token_expires_at
        ):
            return _wechat_cached_access_token

        if not _WECHAT_APPID or not _WECHAT_APPSECRET:
            raise RuntimeError("缺少 WECHAT_APPID/WECHAT_APPSECRET，无法调用微信客服接口")

        query = urllib.parse.urlencode({
            "grant_type": "client_credential",
            "appid": _WECHAT_APPID,
            "secret": _WECHAT_APPSECRET,
        })
        url = f"{_WECHAT_TOKEN_API}?{query}"
        with urllib.request.urlopen(url, timeout=_WECHAT_CUSTOM_REPLY_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError(f"获取微信 access_token 失败: {data}")

        expires_in = int(data.get("expires_in") or 7200)
        _wechat_cached_access_token = str(access_token)
        _wechat_cached_access_token_expires_at = (
            time.time() + max(60, expires_in - _WECHAT_TOKEN_EXPIRE_SKEW_SECONDS)
        )
        return _wechat_cached_access_token


def _post_wechat_customer_text(openid: str, content: str, *, force_refresh_token: bool = False) -> dict:
    access_token = _get_wechat_access_token(force_refresh=force_refresh_token)
    url = f"{_WECHAT_CUSTOM_SEND_API}?access_token={urllib.parse.quote(access_token)}"
    payload = {
        "touser": openid,
        "msgtype": "text",
        "text": {"content": content},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_WECHAT_CUSTOM_REPLY_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _split_text_by_bytes(text: str, max_bytes: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))
        if current and current_bytes + ch_bytes > max_bytes:
            parts.append("".join(current))
            current = []
            current_bytes = 0
        current.append(ch)
        current_bytes += ch_bytes
    if current:
        parts.append("".join(current))
    return parts or [""]


def _send_wechat_customer_text(openid: str, content: str) -> None:
    if not openid:
        raise RuntimeError("缺少微信 openid，无法发送客服消息")

    text = content.strip() or "抱歉，暂时没有生成有效回复。"
    for idx, part in enumerate(_split_text_by_bytes(text, _WECHAT_CUSTOM_REPLY_MAX_BYTES), start=1):
        result = _post_wechat_customer_text(openid, part)
        if result.get("errcode") in (40001, 42001):
            result = _post_wechat_customer_text(openid, part, force_refresh_token=True)
        if result.get("errcode") != 0:
            raise RuntimeError(f"微信客服消息发送失败: {result}")
        log.info(
            "wechat customer reply sent openid=%s part=%d bytes=%d",
            openid,
            idx,
            len(part.encode("utf-8")),
        )


def _run_wechat_async_reply(req: ChatRequest, openid: str) -> None:
    t0 = time.perf_counter()
    log.info(
        "wechat async chat started openid=%s use_rewrite=%s message_len=%d preview=%r",
        openid,
        req.use_rewrite,
        len(req.message),
        preview_text(req.message, 80),
    )
    try:
        resp = _chat_response(req)
    except Exception:
        log.exception("wechat async answer_question failed openid=%s", openid)
        try:
            _send_wechat_customer_text(openid, "抱歉，刚才的问题暂时处理失败，请稍后再试。")
        except Exception:
            log.exception("wechat async failure notice failed openid=%s", openid)
        return

    try:
        _send_wechat_customer_text(openid, resp.answer)
        log.info(
            "wechat async chat completed openid=%s elapsed=%.3fs answer_len=%d",
            openid,
            time.perf_counter() - t0,
            len(resp.answer),
        )
    except Exception:
        log.exception("wechat customer reply failed openid=%s", openid)


def _release_wechat_reply_slot(_future) -> None:
    try:
        _wechat_reply_slots.release()
    except ValueError:
        log.exception("wechat async reply slot release failed")


def _send_wechat_busy_notice(openid: str) -> None:
    try:
        _send_wechat_customer_text(openid, _WECHAT_BUSY_MESSAGE)
        log.info("wechat busy notice sent openid=%s", openid)
    except Exception:
        log.exception("wechat busy notice failed openid=%s", openid)


def _try_submit_wechat_async_reply(req: ChatRequest, openid: str) -> bool:
    if not _wechat_reply_slots.acquire(blocking=False):
        log.warning(
            "wechat async chat rejected openid=%s workers=%d queue_size=%d message_len=%d",
            openid,
            _WECHAT_CUSTOM_REPLY_WORKERS,
            _WECHAT_CUSTOM_REPLY_QUEUE_SIZE,
            len(req.message),
        )
        threading.Thread(
            target=_send_wechat_busy_notice,
            args=(openid,),
            name="wechat-busy-notice",
            daemon=True,
        ).start()
        return False

    try:
        future = _wechat_reply_executor.submit(_run_wechat_async_reply, req, openid)
    except Exception:
        _wechat_reply_slots.release()
        log.exception("wechat async chat submit failed openid=%s", openid)
        threading.Thread(
            target=_send_wechat_busy_notice,
            args=(openid,),
            name="wechat-busy-notice",
            daemon=True,
        ).start()
        return False

    future.add_done_callback(_release_wechat_reply_slot)
    return True


def handle_chat_request(
    req: ChatRequest,
    *,
    wechat_openid: str | None = None,
    async_wechat_reply: bool = False,
) -> ChatResponse:
    if async_wechat_reply:
        openid = (wechat_openid or "").strip()
        if not openid:
            log.warning("wechat async reply requested without openid")
            return _empty_chat_response()

        queued_req = ChatRequest(
            message=req.message,
            history=req.history,
            use_rewrite=req.use_rewrite,
        )
        submitted = _try_submit_wechat_async_reply(queued_req, openid)
        log.info(
            "wechat async chat queued=%s openid=%s workers=%d queue_size=%d message_len=%d",
            submitted,
            openid,
            _WECHAT_CUSTOM_REPLY_WORKERS,
            _WECHAT_CUSTOM_REPLY_QUEUE_SIZE,
            len(req.message),
        )
        return _empty_chat_response()

    return _chat_response(req)


def iter_stream_chat_events(req: ChatRequest) -> Iterator[dict]:
    date_resp = _date_chat_response(req.message)
    if date_resp is not None:
        yield {
            "type": "meta",
            "sources": [],
            "rewritten_query": None,
            "standalone_query": None,
        }
        yield {"type": "delta", "content": date_resp.answer}
        yield {"type": "done", "answer": date_resp.answer, "sources": []}
        return

    hist = _history_for(req)
    log.info(
        "chat_stream request use_rewrite=%s history=%d message_len=%d preview=%r",
        req.use_rewrite,
        len(hist),
        len(req.message),
        preview_text(req.message, 80),
    )
    try:
        yield from stream_answer_question(
            req.message.strip(),
            history=hist,
            use_rewrite=req.use_rewrite,
        )
    except Exception as exc:
        log.exception("chat_stream endpoint error: %s", exc)
        yield {"type": "error", "detail": str(exc)}
