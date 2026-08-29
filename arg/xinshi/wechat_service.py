from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from xml.etree import ElementTree

from arg.xinshi.chat_service import handle_chat_request
from arg.xinshi.logutil import preview_text
from arg.xinshi.schemas import ChatRequest
from arg.xinshi.wechat_crypto import (
    decrypt_text_payload,
    verify_msg_signature,
)

log = logging.getLogger(__name__)


class WeChatPayloadError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class WeChatSignatureError(WeChatPayloadError):
    def __init__(self):
        super().__init__("微信 msg_signature 校验失败", 403)


@dataclass(frozen=True)
class WeChatEnvelope:
    payload: dict
    body_format: str


def _json_for_log(payload: dict | str | None) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


def _cdata(text: str) -> str:
    return f"<![CDATA[{text.replace(']]>', ']]]]><![CDATA[>')}]]>"


def _xml_to_dict(xml_text: str) -> dict:
    try:
        root = ElementTree.fromstring(xml_text.encode("utf-8"))
    except ElementTree.ParseError as exc:
        raise WeChatPayloadError("XML 报文格式错误") from exc

    return {
        child.tag: (child.text or "").strip()
        for child in list(root)
    }


def _dict_to_wechat_xml(payload: dict) -> str:
    parts = ["<xml>"]
    for key, value in payload.items():
        if key == "TimeStamp":
            parts.append(f"<{key}>{int(value)}</{key}>")
        else:
            parts.append(f"<{key}>{_cdata(str(value))}</{key}>")
    parts.append("</xml>")
    return "".join(parts)


def parse_encrypted_envelope(raw_body: bytes, content_type: str | None) -> WeChatEnvelope:
    raw_text = raw_body.decode("utf-8", errors="replace").strip()
    log.info(
        "wechat request raw body len=%d content_type=%r preview=%r",
        len(raw_body),
        content_type,
        raw_text[:1000],
    )
    if not raw_text:
        raise WeChatPayloadError("请求体不能为空")

    if raw_text.startswith("<"):
        payload = _xml_to_dict(raw_text)
        return WeChatEnvelope(payload=payload, body_format="xml")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        log.warning("wechat request body is neither valid XML nor JSON: %s", exc)
        raise WeChatPayloadError("请求体不是合法 XML/JSON") from exc
    if not isinstance(payload, dict):
        log.warning("wechat request JSON body is not object: type=%s", type(payload).__name__)
        raise WeChatPayloadError("请求体 JSON 必须是对象")
    return WeChatEnvelope(payload=payload, body_format="json")


def render_encrypted_envelope(payload: dict, body_format: str) -> str | dict:
    if body_format == "xml":
        return _dict_to_wechat_xml(payload)
    return payload


def _extract_encrypt(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise WeChatPayloadError("请求体必须是对象结构")
    encrypt = payload.get("Encrypt")
    if not isinstance(encrypt, str) or not encrypt:
        raise WeChatPayloadError("请求体缺少 Encrypt")
    return encrypt


def _parse_decrypted_message(plaintext: str) -> tuple[dict, str]:
    text = plaintext.strip()
    if text.startswith("<"):
        return _xml_to_dict(text), "xml"

    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WeChatPayloadError("微信密文明文不是合法 XML/JSON") from exc
    if not isinstance(msg, dict):
        raise WeChatPayloadError("微信密文明文 JSON 必须是对象")
    return msg, "json"


def _text_reply_xml(message: dict) -> str:
    return _dict_to_wechat_xml(message)


def _text_reply_json(message: dict) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def _reply_plaintext(message: dict, message_format: str) -> str:
    if message_format == "xml":
        return _text_reply_xml(message)
    return _text_reply_json(message)


def build_encrypted_chat_reply(
    payload: dict,
    *,
    msg_signature: str,
    timestamp: str,
    nonce: str,
    openid: str | None,
    signature_present: bool,
) -> dict | None:
    log.info("wechat request encrypted payload=%s", _json_for_log(payload))
    encrypt = _extract_encrypt(payload)
    if not verify_msg_signature(msg_signature, timestamp, nonce, encrypt):
        log.warning(
            "invalid WeChat msg_signature timestamp=%s nonce=%s openid=%s signature_ignored=%s",
            timestamp,
            nonce,
            openid,
            signature_present,
        )
        raise WeChatSignatureError()

    plaintext = decrypt_text_payload(encrypt)
    log.info("wechat request decrypted plaintext=%s", plaintext)
    msg, message_format = _parse_decrypted_message(plaintext)
    log.info("wechat request decrypted payload=%s", _json_for_log(msg))
    msg_type = str(msg.get("MsgType") or "").strip()
    content = str(msg.get("Content") or "").strip()
    if msg_type and msg_type != "text":
        log.info("ignore unsupported WeChat msg_type=%s openid=%s", msg_type, openid)
        log.info("wechat response plaintext payload=%s", _json_for_log("success"))
        log.info("wechat response encrypted payload=%s", _json_for_log(None))
        return None
    if not content:
        log.info("ignore WeChat message without Content msg_type=%s openid=%s", msg_type, openid)
        log.info("wechat response plaintext payload=%s", _json_for_log("success"))
        log.info("wechat response encrypted payload=%s", _json_for_log(None))
        return None

    log.info(
        "wechat encrypted message received format=%s from=%s to=%s openid=%s content_len=%d preview=%r",
        message_format,
        msg.get("FromUserName"),
        msg.get("ToUserName"),
        openid,
        len(content),
        preview_text(content, 80),
    )
    reply_openid = str(msg.get("FromUserName") or openid or "").strip()
    handle_chat_request(
        ChatRequest(message=content),
        wechat_openid=reply_openid,
        async_wechat_reply=True,
    )
    log.info("wechat response plaintext payload=%s", _json_for_log(""))
    log.info("wechat response encrypted payload=%s", _json_for_log(None))
    return None
