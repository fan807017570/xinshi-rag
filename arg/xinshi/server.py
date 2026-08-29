"""
新实中学招生顾问 RAG — HTTP 服务。

启动方式（任选其一，均需已配置 Milvus / 模型）：

    cd /path/to/xinshi-rag
    python -m uvicorn arg.xinshi.server:app --host 0.0.0.0 --port 8765

    # 或在 PyCharm 里直接运行本文件（会内嵌启动 uvicorn）：
    python arg/xinshi/server.py

浏览器打开：http://127.0.0.1:8765/
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from importlib.util import find_spec
from pathlib import Path

# 保证可解析包 arg.xinshi
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 本地开发时自动加载 arg/xinshi/.env（Docker 环境中 .env 不存在，此行无副作用）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from arg.xinshi.chat_service import (
    handle_chat_request,
    is_non_natural_language_message,
    iter_stream_chat_events,
)
from arg.xinshi.logutil import configure_logging
from arg.xinshi.schemas import ChatRequest, ChatResponse
from arg.xinshi.wechat_crypto import WeChatCryptoError, verify_url_signature
from arg.xinshi.wechat_service import (
    WeChatPayloadError,
    WeChatSignatureError,
    build_encrypted_chat_reply,
    parse_encrypted_envelope,
    render_encrypted_envelope,
)

configure_logging()
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 允许上传的文件后缀
_ALLOWED_SUFFIXES = {".md", ".txt"}
# 安全文件名：只允许字母、数字、连字符、下划线、点
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

app = FastAPI(title="新实中学招生顾问", version="1.0")
_HAS_PYTHON_MULTIPART = find_spec("multipart") is not None


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        t0 = time.perf_counter()
        log.info("HTTP %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            log.info(
                "HTTP %s %s -> %s in %.3fs",
                request.method,
                request.url.path,
                response.status_code,
                time.perf_counter() - t0,
            )
            return response
        except Exception:
            log.exception(
                "HTTP %s %s failed after %.3fs",
                request.method,
                request.url.path,
                time.perf_counter() - t0,
            )
            raise
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_wechat_message(
    request: Request,
    msg_signature: str = Query(..., description="微信消息体签名"),
    timestamp: str = Query(..., description="时间戳"),
    nonce: str = Query(..., description="随机数"),
    signature: str | None = Query(None, description="微信 URL 签名，POST 加密消息不使用"),
    openid: str | None = Query(None, description="微信 openid"),
    encrypt_type: str | None = Query(None, description="加密类型"),
):
    if encrypt_type and encrypt_type != "aes":
        raise HTTPException(status_code=400, detail="仅支持 encrypt_type=aes")

    raw_body = await request.body()
    try:
        envelope = parse_encrypted_envelope(raw_body, request.headers.get("content-type"))
        encrypted_reply = build_encrypted_chat_reply(
            envelope.payload,
            msg_signature=msg_signature,
            timestamp=timestamp,
            nonce=nonce,
            openid=openid,
            signature_present=bool(signature),
        )
    except WeChatSignatureError:
        return PlainTextResponse("", status_code=403)
    except (WeChatCryptoError, WeChatPayloadError) as exc:
        log.warning("wechat request rejected: status=%s detail=%s", exc.status_code, exc.detail)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as e:
        log.exception("wechat message chat dispatch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    if encrypted_reply is None:
        return PlainTextResponse("")
    rendered_reply = render_encrypted_envelope(encrypted_reply, envelope.body_format)
    if envelope.body_format == "xml":
        log.info("wechat response encrypted xml=%s", rendered_reply)
        return PlainTextResponse(rendered_reply, media_type="application/xml; charset=utf-8")
    return JSONResponse(content=rendered_reply)


@app.get("/api/chat", response_model=ChatResponse)
async def chat_wechat_validation(
    signature: str = Query(..., description="微信加密签名"),
    timestamp: str = Query(..., description="时间戳"),
    nonce: str = Query(..., description="随机数"),
    echostr: str = Query(..., description="随机字符串"),
):
    if not verify_url_signature(signature, timestamp, nonce):
        log.warning(
            "invalid WeChat signature timestamp=%s nonce=%s echostr_len=%d",
            timestamp,
            nonce,
            len(echostr),
        )
        return PlainTextResponse("", status_code=403)

    log.info("valid WeChat signature; dispatch echostr to chat handler len=%d", len(echostr))
    if is_non_natural_language_message(echostr):
        log.info("return non-natural WeChat echostr directly")
        return PlainTextResponse(echostr)

    try:
        return handle_chat_request(ChatRequest(message=echostr))
    except Exception as e:
        log.exception("wechat validation chat dispatch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _iter_events():
        for item in iter_stream_chat_events(req):
            yield _sse(item)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _iter_events(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers,
    )


@app.get("/api/docs", summary="列出 data/docs 目录下的所有文件")
async def list_docs():
    files = sorted(
        [
            {"name": f.name, "size": f.stat().st_size, "suffix": f.suffix}
            for f in DOCS_DIR.iterdir()
            if f.is_file()
        ],
        key=lambda x: x["name"],
    )
    return {"files": files, "dir": str(DOCS_DIR)}


if _HAS_PYTHON_MULTIPART:

    @app.post("/api/docs/upload", summary="上传文档到 data/docs 目录")
    async def upload_doc(
        file: UploadFile = File(...),
        reindex: bool = False,
    ):
        filename = (file.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型 {suffix}，仅允许：{', '.join(_ALLOWED_SUFFIXES)}",
            )
        if not _SAFE_FILENAME_RE.match(filename):
            raise HTTPException(status_code=400, detail="文件名包含非法字符")

        dest = DOCS_DIR / filename
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10 MB 上限
            raise HTTPException(status_code=400, detail="文件大小不能超过 10 MB")

        dest.write_bytes(content)
        log.info("uploaded doc: %s (%d bytes) reindex=%s", dest, len(content), reindex)

        result: dict = {"saved": filename, "bytes": len(content), "reindex": reindex}

        if reindex:
            try:
                from arg.xinshi.ingest import ingest as _ingest
                from arg.xinshi.application import reload_vectorstore

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _ingest)
                await loop.run_in_executor(None, reload_vectorstore)
                result["reindex_status"] = "ok"
                log.info("reindex + vectorstore reload triggered by upload of %s", filename)
            except Exception as exc:
                log.exception("reindex failed after upload of %s: %s", filename, exc)
                result["reindex_status"] = f"failed: {exc}"

        return JSONResponse(content=result)
else:

    @app.post("/api/docs/upload", summary="上传文档到 data/docs 目录")
    async def upload_doc():
        raise HTTPException(
            status_code=503,
            detail="当前环境缺少 python-multipart，上传接口已禁用",
        )


# 全局重建任务状态，防止并发重复触发
_reindex_lock = asyncio.Lock()
_reindex_status: dict = {"running": False, "last_result": None}


@app.post("/api/docs/reindex", summary="触发全量重建向量索引")
async def trigger_reindex():
    if _reindex_status["running"]:
        raise HTTPException(status_code=409, detail="索引重建任务正在运行，请稍后再试")

    async def _run():
        _reindex_status["running"] = True
        t0 = time.perf_counter()
        try:
            from arg.xinshi.ingest import ingest as _ingest
            from arg.xinshi.application import reload_vectorstore

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _ingest)
            elapsed = round(time.perf_counter() - t0, 2)
            log.info("reindex finished in %.2fs, reloading vectorstore...", elapsed)

            # 关键：刷新内存中的 vectorstore，否则仍查旧数据
            await loop.run_in_executor(None, reload_vectorstore)

            _reindex_status["last_result"] = {"status": "ok", "elapsed_s": elapsed}
            log.info("vectorstore reloaded, reindex fully complete")
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            _reindex_status["last_result"] = {"status": "error", "detail": str(exc), "elapsed_s": elapsed}
            log.exception("reindex failed after %.2fs: %s", elapsed, exc)
        finally:
            _reindex_status["running"] = False

    asyncio.create_task(_run())
    return JSONResponse(content={"accepted": True, "message": "索引重建任务已启动，请稍后查询状态"})


@app.get("/api/docs/reindex/status", summary="查询索引重建任务状态")
async def reindex_status():
    return JSONResponse(content={
        "running": _reindex_status["running"],
        "last_result": _reindex_status["last_result"],
    })


@app.get("/")
async def index_page():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(index, media_type="text/html; charset=utf-8")


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("XINSHI_HOST", "0.0.0.0")
    port = int(os.environ.get("XINSHI_PORT", "8765"))
    log.info("Starting uvicorn on http://%s:%s (XINSHI_LOG_LEVEL=%s)", host, port, os.environ.get("XINSHI_LOG_LEVEL", "INFO"))
    uvicorn.run(app, host=host, port=port)
