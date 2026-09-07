from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass

try:  # pragma: no cover - optional dependency path
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except Exception:  # pragma: no cover - optional dependency path
    Cipher = algorithms = modes = None  # type: ignore[assignment]

WECHAT_TOKEN_ENV = "WECHAT_TOKEN"
WECHAT_ENCODING_AES_KEY_ENV = "WECHAT_ENCODING_AES_KEY"
WECHAT_APPID_ENV = "WECHAT_APPID"

WECHAT_AES_BLOCK_SIZE = 32


class WeChatCryptoError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class WeChatConfig:
    token: str
    encoding_aes_key: str
    appid: str

    @classmethod
    def from_env(cls) -> "WeChatConfig":
        return cls(
            token=os.environ.get(WECHAT_TOKEN_ENV, "").strip(),
            encoding_aes_key=os.environ.get(WECHAT_ENCODING_AES_KEY_ENV, "").strip(),
            appid=os.environ.get(WECHAT_APPID_ENV, "").strip(),
        )

    def validate(self) -> None:
        if not self.token or not self.encoding_aes_key or not self.appid:
            raise WeChatCryptoError(
                "WECHAT_TOKEN/WECHAT_ENCODING_AES_KEY/WECHAT_APPID 配置不完整",
                500,
            )
        self.aes_key

    @property
    def aes_key(self) -> bytes:
        if not self.encoding_aes_key:
            raise WeChatCryptoError("WECHAT_ENCODING_AES_KEY 未配置", 500)
        try:
            aes_key = base64.b64decode(self.encoding_aes_key + "=", validate=True)
        except Exception as exc:
            raise WeChatCryptoError("WECHAT_ENCODING_AES_KEY 配置错误", 500) from exc
        if len(aes_key) != 32:
            raise WeChatCryptoError("WECHAT_ENCODING_AES_KEY 解码后必须为 32 字节", 500)
        return aes_key


def url_signature(token: str, timestamp: str, nonce: str) -> str:
    raw = "".join(sorted([token, timestamp, nonce]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def verify_url_signature(signature: str, timestamp: str, nonce: str, config: WeChatConfig | None = None) -> bool:
    config = config or WeChatConfig.from_env()
    if not config.token:
        return False
    return hmac.compare_digest(url_signature(config.token, timestamp, nonce), signature)


def msg_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    raw = "".join(sorted([token, timestamp, nonce, encrypt]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def verify_msg_signature(
    msg_signature_value: str,
    timestamp: str,
    nonce: str,
    encrypt: str,
    config: WeChatConfig | None = None,
) -> bool:
    config = config or WeChatConfig.from_env()
    if not config.token:
        return False
    expected = msg_signature(config.token, timestamp, nonce, encrypt)
    return hmac.compare_digest(expected, msg_signature_value)


def _pkcs7_pad(data: bytes) -> bytes:
    amount = WECHAT_AES_BLOCK_SIZE - (len(data) % WECHAT_AES_BLOCK_SIZE)
    if amount == 0:
        amount = WECHAT_AES_BLOCK_SIZE
    return data + bytes([amount]) * amount


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise WeChatCryptoError("微信密文解密后为空")

    amount = data[-1]
    if amount < 1 or amount > WECHAT_AES_BLOCK_SIZE:
        raise WeChatCryptoError("微信密文 PKCS#7 填充错误")
    if data[-amount:] != bytes([amount]) * amount:
        raise WeChatCryptoError("微信密文 PKCS#7 填充不一致")
    return data[:-amount]


def _aes_decrypt(encrypt: str, config: WeChatConfig) -> bytes:
    if Cipher is None or algorithms is None or modes is None:
        raise WeChatCryptoError("cryptography 未安装，无法处理微信加密消息", 500)
    try:
        encrypted = base64.b64decode(encrypt, validate=True)
    except Exception as exc:
        raise WeChatCryptoError("Encrypt 不是合法 Base64") from exc

    aes_key = config.aes_key
    decryptor = Cipher(algorithms.AES(aes_key), modes.CBC(aes_key[:16])).decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    return _pkcs7_unpad(padded)


def _aes_encrypt(plain: bytes, config: WeChatConfig) -> str:
    if Cipher is None or algorithms is None or modes is None:
        raise WeChatCryptoError("cryptography 未安装，无法处理微信加密消息", 500)
    aes_key = config.aes_key
    encryptor = Cipher(algorithms.AES(aes_key), modes.CBC(aes_key[:16])).encryptor()
    encrypted = encryptor.update(_pkcs7_pad(plain)) + encryptor.finalize()
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_text_payload(encrypt: str, config: WeChatConfig | None = None) -> str:
    config = config or WeChatConfig.from_env()
    full = _aes_decrypt(encrypt, config)
    if len(full) < 20:
        raise WeChatCryptoError("微信密文明文长度不足")

    msg_len = int.from_bytes(full[16:20], "big")
    msg_start = 20
    msg_end = msg_start + msg_len
    if msg_end > len(full):
        raise WeChatCryptoError("微信密文 msg_len 越界")

    msg_bytes = full[msg_start:msg_end]
    appid = full[msg_end:].decode("utf-8", errors="replace")
    if appid != config.appid:
        raise WeChatCryptoError("微信 Appid 不匹配", 403)

    return msg_bytes.decode("utf-8")


def decrypt_json_payload(encrypt: str, config: WeChatConfig | None = None) -> dict:
    msg_text = decrypt_text_payload(encrypt, config)
    try:
        msg = json.loads(msg_text)
    except json.JSONDecodeError as exc:
        raise WeChatCryptoError("微信密文明文不是合法 JSON") from exc

    if not isinstance(msg, dict):
        raise WeChatCryptoError("微信密文明文 JSON 必须是对象")
    return msg


def encrypt_text_payload(
    message_text: str,
    timestamp: str,
    nonce: str,
    config: WeChatConfig | None = None,
    random_bytes: bytes | None = None,
) -> dict:
    config = config or WeChatConfig.from_env()
    random_part = random_bytes or secrets.token_bytes(16)
    if len(random_part) != 16:
        raise WeChatCryptoError("微信加密 random 必须为 16 字节", 500)

    msg = message_text.encode("utf-8")
    full = random_part + len(msg).to_bytes(4, "big") + msg + config.appid.encode("utf-8")
    encrypted = _aes_encrypt(full, config)
    try:
        timestamp_value = int(timestamp)
    except ValueError as exc:
        raise WeChatCryptoError("TimeStamp 必须为整数") from exc
    return {
        "Encrypt": encrypted,
        "MsgSignature": msg_signature(config.token, timestamp, nonce, encrypted),
        "TimeStamp": timestamp_value,
        "Nonce": nonce,
    }


def encrypt_json_payload(
    message: dict,
    timestamp: str,
    nonce: str,
    config: WeChatConfig | None = None,
    random_bytes: bytes | None = None,
) -> dict:
    msg = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return encrypt_text_payload(msg, timestamp, nonce, config, random_bytes)
