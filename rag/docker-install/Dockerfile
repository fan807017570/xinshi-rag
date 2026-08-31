# =============================================================================
# 新实中学招生顾问 RAG 服务 Dockerfile
# 构建上下文：xinshi-rag/ 目录
#   docker build -t xinshi-rag -f rag/docker-install/Dockerfile .
# =============================================================================

FROM python:3.11-slim

# 防止交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace 缓存目录（挂载宿主机目录可避免每次重新下载模型）
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY rag/requirements-common.txt /app/requirements-common.txt

# 安装 Python 依赖
# 安装 PyTorch CPU-only wheel。
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0+cpu" && \
    pip install --no-cache-dir -r requirements-common.txt

# 复制默认配置和完整 RAG 应用包。运行时 Compose 会用宿主机 config/
# 目录覆盖 /app/config，因此修改配置后无需重新构建镜像。
COPY config/ /app/config/
COPY rag/ /app/rag/

# 宿主机资料挂载点：
#   - /app/config：外部运行配置
#   - /app/rag/data/base：ingest.py 读取知识库 Markdown
#   - /app/rag/data/docs：Web 上传/列举文档
# 运行示例：
#   docker run --rm -p 18765:8765 \
#     -v /宿主机/知识库目录:/app/rag/data/base:ro \
#     -v /宿主机/上传目录:/app/rag/data/docs \
#     xinshi-rag:latest
RUN mkdir -p /app/config /app/rag/data/base /app/rag/data/docs
VOLUME ["/app/config", "/app/rag/data/base", "/app/rag/data/docs"]

# 确保包可被 Python 发现
ENV PYTHONPATH=/app

# 服务端口
EXPOSE 8765

# 启动命令（支持通过 XINSHI_HOST / XINSHI_PORT 环境变量覆盖）
CMD ["python", "-m", "uvicorn", "rag.server:app", "--host", "0.0.0.0", "--port", "8765"]
