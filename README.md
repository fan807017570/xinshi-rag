# xinshi-rag

新实中学 RAG 服务，提供知识库文档检索、重排、LLM 问答、文档管理和微信消息接入能力。

项目支持两种应用部署方式：

- Docker 部署：应用以 Docker 容器运行。
- macOS / Ubuntu 原生部署：应用安装到 Python 虚拟环境后直接运行。

项目统一使用 CPU 版本 PyTorch，不使用 GPU，因此服务器不需要安装 NVIDIA 驱动、CUDA Toolkit 或 cuDNN。

Milvus、etcd 和 MinIO 已从应用编排中拆出，由 `install/milvus/` 独立管理，两种应用部署方式均可复用这套基础设施。

应用源码现在直接位于根目录 `rag/`，Python 包名为 `rag`，不再包含 `arg` 中间目录。服务入口为 `rag/server.py`，Uvicorn 模块入口为 `rag.server:app`。

## 服务结构

```text
浏览器 / 微信
      │
      ▼
 xinshi-rag ──────────────► DeepSeek API
      │
      ▼
    Milvus
    ├── etcd
    └── MinIO
```

Docker 应用通过共享网络访问 `milvus-standalone:19530`；macOS / Ubuntu 原生应用默认通过 `127.0.0.1:19531` 访问 Milvus。

## 目录结构

```text
xinshi-rag/
├── config/
│   └── application.ini             # 可外部修改的应用运行配置
├── install/
│   ├── install-docker.sh           # Ubuntu Docker 安装
│   ├── install-ubuntu-deps.sh      # Ubuntu 原生依赖安装
│   ├── macos/
│   │   ├── install-deps.sh         # macOS 依赖检查/安装
│   │   └── deploy-components.sh    # macOS 基础组件 Docker 部署
│   └── milvus/
│       ├── docker-compose.yml      # Milvus、etcd、MinIO
│       ├── install.sh              # 基础设施管理
│       └── .env.example
├── rag/
│   ├── server.py                    # FastAPI/Uvicorn 服务入口
│   ├── application.py               # RAG 问答流程
│   ├── ingest.py                    # 知识库索引构建
│   ├── retriever.py                 # Milvus 向量检索
│   ├── reranker.py                  # 检索结果重排
│   ├── data/                       # 基础知识库和上传文档
│   ├── static/                     # Web 静态页面
│   ├── requirements-common.txt
│   ├── requirements.txt            # 默认 CPU 依赖
│   ├── .env.example
│   ├── docker-install/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml      # 仅包含 RAG 应用
│   │   ├── build.sh                # Docker 镜像构建
│   │   ├── deploy.sh               # Docker 应用部署
│   │   ├── build-macos.sh          # macOS Docker 镜像构建
│   │   └── deploy-macos.sh         # macOS Docker 应用部署
│   └── build/
│       ├── requirements.txt        # Ubuntu CPU 依赖
│       ├── build-ubuntu.sh         # Ubuntu 原生环境构建
│       ├── run-ubuntu.sh           # Ubuntu 原生应用启动
│       ├── requirements-macos.txt  # macOS CPU 依赖
│       ├── build-macos.sh          # macOS 原生环境构建
│       └── run-macos.sh            # macOS 原生应用启动
├── CHANGELOG.md
└── README.md
```

## 部署前准备

所有命令默认在项目根目录执行：

```bash
cd /path/to/xinshi-rag
```

部署前需要准备：

- 可访问 DeepSeek API 的网络和有效 API Key。
- Embedding 模型 `bge-base-zh-v1.5`。
- Reranker 模型 `bge-reranker-large`。
- Docker 部署需要 Docker Engine 和 Docker Compose Plugin。
- macOS 部署需要 Docker Desktop；原生部署还需要 Homebrew 和 Python 3.10+（推荐 3.11）。
- Ubuntu 原生部署需要 Python 3、venv 和基础编译工具。
- 不需要安装 CUDA、cuDNN 或其他 GPU 运行时。

默认模型目录为：

```text
./rag/models/bge-base-zh-v1.5
./rag/models/bge-reranker-large
```

也可以在应用 `.env` 中配置其他绝对路径。

## 第一步：安装 Docker

### macOS

Docker 部署使用 Docker Desktop。安装 Homebrew 后，可执行以下脚本检查并安装 Python 3.11 和 Docker Desktop：

```bash
bash install/macos/install-deps.sh --install
```

如果只部署 Docker 服务、不做原生 Python 部署：

```bash
bash install/macos/install-deps.sh --install --docker-only
```

### Ubuntu

```bash
bash install/install-docker.sh
```

脚本会安装 Docker Engine、Buildx 和 Docker Compose Plugin，并将当前用户加入 `docker` 用户组。安装后需要重新登录，或者执行：

```bash
newgrp docker
```

如果服务器已经安装 Docker 和 `docker compose`，可以跳过此步骤。

## 第二步：部署 Milvus 基础设施

首次启动（macOS）：

```bash
bash install/macos/deploy-components.sh up
```

Linux 或通用方式：

```bash
bash install/milvus/install.sh up
```

首次执行会自动将 `install/milvus/.env.example` 复制为 `install/milvus/.env`。默认端口如下：

| 服务 | 宿主机地址 | 用途 |
| --- | --- | --- |
| Milvus | `localhost:19531` | macOS / Ubuntu 原生应用连接地址 |
| MinIO API | `http://localhost:19000` | Milvus 对象存储 API |
| MinIO 控制台 | `http://localhost:19001` | MinIO 管理页面 |

MinIO 默认账号和密码均为 `minioadmin`，生产环境部署前应修改为安全凭据。

常用运维命令：

```bash
# 查看状态
bash install/milvus/install.sh status

# 查看全部日志
bash install/milvus/install.sh logs

# 查看 Milvus 日志
bash install/milvus/install.sh logs milvus-standalone

# 重启
bash install/milvus/install.sh restart

# 停止容器，保留数据卷
bash install/milvus/install.sh down
```

默认配置沿用旧一体化 Compose 的项目名和数据卷名：

```text
xinshi_etcd_data
xinshi_minio_data
xinshi_milvus_data
```

这样拆分后仍可继续使用已有 Milvus 数据。

## 第三步：配置应用

`.env` 只保存 API Key、端口和 Docker 宿主机挂载路径。首次部署先复制模板：

```bash
cp rag/.env.example rag/.env
```

至少需要修改：

```dotenv
DEEPSEEK_API_KEY=你的真实API密钥
```

`config.py` 中原有的运行参数已全部移到 `config/application.ini`，包括：

| 配置段 | 主要配置 | 说明 |
| --- | --- | --- |
| `[milvus]` | `enabled`、`host`、`port`、`collection` | 原生运行的 Milvus 配置 |
| `[milvus:docker]` | `host`、`port` | Docker profile 覆盖项 |
| `[models]` | `embedding_model`、`rerank_model`、缓存目录 | 模型及缓存路径 |
| `[reranker]` | backend、batch、长度和缓存限制 | 重排器参数 |
| `[retrieval]` | `top_k_retrieve`、`top_k_rerank`、角色倍率 | 检索参数 |
| `[conversation]` | `max_history_messages` | 多轮会话历史条数 |

原生部署默认使用 `[runtime] profile = native`；Docker Compose 自动选择 `docker` profile，并把宿主机 `config/` 目录只读挂载到容器 `/app/config`。

配置优先级为“环境变量 > profile 配置段 > 通用配置段”。旧环境变量仍兼容；如果 `.env` 中保留了 `MILVUS_HOST`、`TOP_K_RETRIEVE` 等旧配置，它们会覆盖 INI 文件。

### 修改配置后重启

Docker 部署直接重启应用，不需要 `--build`：

```bash
# macOS
bash rag/docker-install/deploy-macos.sh --restart

# Linux / 通用
bash rag/docker-install/deploy.sh --restart
```

原生部署使用 `Ctrl+C` 停止后，重新执行对应启动脚本：

```bash
bash rag/build/run-macos.sh
# 或 bash rag/build/run-ubuntu.sh
```

## 部署方式一：Docker

确保 Milvus 已启动。macOS 使用：

```bash
bash install/macos/deploy-components.sh status
```

Linux 或通用方式使用：

```bash
bash install/milvus/install.sh status
```

macOS 下构建并启动应用：

```bash
bash rag/docker-install/build-macos.sh
bash rag/docker-install/deploy-macos.sh
```

重新构建并启动：

```bash
bash rag/docker-install/deploy-macos.sh --build
```

Intel Mac 和 Apple Silicon 默认使用 `linux/amd64`；如确认镜像支持 ARM64，可覆盖：

```bash
DOCKER_PLATFORM=linux/arm64 bash install/macos/deploy-components.sh up
DOCKER_PLATFORM=linux/arm64 bash rag/docker-install/deploy-macos.sh --build
```

构建应用镜像：

```bash
bash rag/docker-install/build.sh
```

带版本号构建：

```bash
bash rag/docker-install/build.sh 1.0.0
```

构建并推送到镜像仓库：

```bash
REGISTRY=registry.example.com/xinshi \
  bash rag/docker-install/build.sh 1.0.0 --push
```

启动应用：

```bash
bash rag/docker-install/deploy.sh
```

重新构建并启动：

```bash
bash rag/docker-install/deploy.sh --build
```

其他运维命令：

```bash
# 重启应用
bash rag/docker-install/deploy.sh --restart

# 停止应用，保留数据卷和 Milvus 基础设施
bash rag/docker-install/deploy.sh --down

# 查看应用状态
docker compose --env-file rag/.env \
  -f rag/docker-install/docker-compose.yml ps

# 查看应用日志
docker compose --env-file rag/.env \
  -f rag/docker-install/docker-compose.yml logs -f xinshi-rag
```

Docker 应用依赖名为 `xinshi-net` 的共享网络。如果启动时提示网络不存在，请先执行：

```bash
bash install/milvus/install.sh up
```

## 部署方式二：macOS / Ubuntu 原生运行

原生方式只将 RAG 应用安装到 Python 虚拟环境；Milvus 仍通过前面独立部署的 Docker 基础设施运行。

### macOS

先启动基础组件：

```bash
bash install/macos/deploy-components.sh up
```

然后构建并启动原生应用：

```bash
cp rag/.env.example rag/.env
# 编辑 rag/.env，至少填写 DEEPSEEK_API_KEY
bash rag/build/build-macos.sh
bash rag/build/run-macos.sh
```

macOS 原生依赖使用 PyPI 上的 `torch==2.6.0`，不会安装 Linux 专用的 `+cpu` wheel；运行时默认连接 `127.0.0.1:19531`。

### Ubuntu

安装系统依赖：

```bash
bash install/install-ubuntu-deps.sh
```

确认应用 `.env` 中包含 API Key：

```dotenv
DEEPSEEK_API_KEY=你的真实API密钥
```

Milvus、模型和检索参数统一修改 `config/application.ini`。

构建 Python 虚拟环境：

```bash
bash rag/build/build-ubuntu.sh
```

该脚本会：

- 在项目根目录创建 `.venv`。
- 升级 pip、setuptools 和 wheel。
- 安装公共依赖。
- 从 PyTorch 官方 CPU 软件源安装 CPU-only PyTorch，不安装 CUDA 组件。
- 检查 FastAPI、PyMilvus、PyTorch 和 Transformers 是否可导入。

启动应用：

```bash
bash rag/build/run-ubuntu.sh
```

脚本内部使用以下模块入口启动服务：

```bash
python -m uvicorn rag.server:app --host 0.0.0.0 --port 18765
```

前台运行时使用 `Ctrl+C` 停止应用。如需长期后台运行，建议后续配置 systemd、Supervisor 或其他进程管理工具。

自定义虚拟环境或 Python 命令：

```bash
VENV_DIR=/opt/xinshi-rag/venv \
PYTHON_BIN=/usr/bin/python3 \
  bash rag/build/build-ubuntu.sh
```

启动时使用相同的虚拟环境路径：

```bash
VENV_DIR=/opt/xinshi-rag/venv \
  bash rag/build/run-ubuntu.sh
```

## 服务访问与健康检查

默认应用地址：

```text
http://服务器地址:18765/
```

健康检查：

```bash
curl http://127.0.0.1:18765/api/health
```

正常响应：

```json
{
  "status": "ok",
  "retriever_backend": "milvus",
  "collection": "Xinshi_school",
  "reranker_backend": "torch",
  "llm_backend": "deepseek",
  "llm_model": "deepseek-v4-flash",
  "llm_rewrite_model": "deepseek-v4-flash"
}
```

当 `[milvus] enabled = true` 时，Milvus 连接或依赖异常会直接导致应用启动失败，
不会自动降级为本地字符串检索。只有显式配置 `enabled = false` 才会启用本地检索。

LLM 同样采用严格配置：缺少 `DEEPSEEK_API_KEY`、参数格式错误或客户端初始化失败时，
健康检查和聊天接口返回 HTTP 503，不会使用本地兜底回答器。例如：

```json
{
  "status": "error",
  "component": "llm",
  "detail": "未配置 DEEPSEEK_API_KEY；请在 rag/.env 或运行环境中设置有效 Key"
}
```

远程 DeepSeek 请求因 Key、模型、账户或网络问题失败时，聊天接口会返回脱敏错误，
不会在错误信息中输出凭据。

查看当前索引实体数和来源文件覆盖率：

```bash
curl http://127.0.0.1:18765/api/docs/index
```

响应中的 `coverage_ratio` 应为 `1.0`，且 `missing_sources` 应为空。

如修改了 `RAG_PORT` 或 `XINSHI_PORT`，请使用修改后的端口。

### 检索来源与分层指标

聊天响应保留原 `sources` 章节名数组，并新增：

- `source_details`：最终 Top-K 的稳定 `document_id`、相对 `source`、`section`、
  原始召回排名/分数和重排排名/分数。
- `retrieval_metrics`：原始召回数、分数过滤数、角色过滤数、实际重排候选数、
  最终结果数，以及检索和重排耗时。
- 请求设置 `include_retrieval_trace=true` 时额外返回 `retrieval_candidates`，
  用于计算原始 Recall@K；默认不返回，避免扩大正常聊天响应。

`[reranker] candidate_limit` 必须不小于
`top_k_retrieve * role_filter_multiplier`。默认配置为 `60`，保证角色查询召回的候选
不会在进入 Reranker 前被截断；配置不足时应用会明确拒绝启动。

### RAG 质量评测

复制并填写 `rag/evaluation.example.jsonl`，其中 `expected_document_ids` 使用
`source_details` 返回的稳定文档 ID。然后执行：

```bash
.venv/bin/python -m rag.evaluation /path/to/evaluation.jsonl
```

评测报告分别输出：

- `raw_recall_at_20`：重排前原始候选的 Recall@20；
- `rerank_hit_at_5`：重排后 Top5 的文档命中率；
- `answer_accuracy`：最终答案是否包含用例标注的全部期望关键词。

质量评测需要正确配置 `DEEPSEEK_API_KEY`，并会调用对应的外部模型服务；配置缺失或
错误时直接失败，不生成兜底答案。

## 常见问题

### 应用提示找不到 `xinshi-net`

Milvus 基础设施尚未启动，执行：

```bash
bash install/milvus/install.sh up
```

### 应用无法连接 Milvus

- Docker 应用应连接 `milvus-standalone:19530`，该配置由应用 Compose 自动设置。
- macOS / Ubuntu 原生应用默认连接 `127.0.0.1:19531`。
- 使用 `bash install/milvus/install.sh status` 检查服务状态。
- 检查 `install/milvus/.env` 和 `config/application.ini` 中的端口是否一致。

### LLM 配置错误或健康检查返回 503

从模板创建本地配置，并填入新生成的有效 Key：

```bash
cp rag/.env.example rag/.env
```

至少配置：

```dotenv
DEEPSEEK_API_KEY=你的新Key
DEEPSEEK_MODEL=deepseek-v4-flash
```

不要把真实 Key 写入源码、`.env.example` 或提交到 Git。修改后必须重启应用进程。

### `ingest.py` 无法导入 Milvus 或 PyMilvus

请从项目根目录使用模块方式运行，避免直接执行源码文件：

```bash
.venv/bin/python -m rag.ingest
```

如果出现 `cannot import name 'Milvus'` 或 `PyMilvus 不可用`，说明当前 Python
环境仍使用旧依赖或没有安装完整依赖。macOS 原生部署重新执行：

```bash
bash rag/build/build-macos.sh
```

如果已经进入项目虚拟环境，也可以只重新安装应用依赖：

```bash
python -m pip install -r rag/build/requirements-macos.txt
```

项目当前使用 Milvus 2.4.4，因此 PyMilvus 固定在 2.4.x。不要单独升级到最新版
`langchain-milvus` / PyMilvus 3.x；如需迁移，应先备份数据并整体升级 Milvus 服务。

### macOS 安装 `cryptography` 时提示 Cargo lock file version 4

Apple Silicon 机器上如果使用 `/usr/local/bin/python3.11`，实际运行的可能是
Rosetta 下的 Intel Python。新版本 `cryptography` 没有适用于该组合的 wheel 时，
pip 会回退到 Rust 源码构建，并可能因 Cargo 版本过低失败。

重新执行 macOS 构建脚本即可。脚本会优先选择 `/opt/homebrew` 的 arm64 Python、
备份架构不匹配的旧虚拟环境，并强制为 `cryptography` 选择二进制 wheel：

```bash
bash rag/build/build-macos.sh
```

构建完成后确认解释器为：

```text
/Users/anranfan/core-bank-dp/xinshi-admin/xinshi-rag/.venv/bin/python
```

构建脚本同时将 setuptools 限制在 81 以下，因为项目使用的 PyMilvus 2.4 仍依赖
已弃用的 `pkg_resources`。

### 模型目录不存在

确认以下目录存在且运行用户有读取权限：

```text
./rag/models/bge-base-zh-v1.5
./rag/models/bge-reranker-large
```

如果模型在其他位置，原生部署修改 `config/application.ini` 的 `[models]` 路径；Docker 部署还需修改 `.env` 中的 `XINSHI_EMBEDDING_MODEL_DIR` 和 `XINSHI_RERANK_MODEL_DIR` 宿主机挂载源。

### Docker 端口冲突

可分别修改：

- 应用 `.env` 中的 `RAG_PORT`。
- `install/milvus/.env` 中的 `MILVUS_EXPOSED_PORT`、`MINIO_API_PORT` 和 `MINIO_UI_PORT`。

修改 Milvus 暴露端口后，macOS / Ubuntu 原生应用的 `[milvus] port` 必须同步修改；Docker 应用使用 `[milvus:docker]` 的容器内部端口，不受宿主机暴露端口影响。

## 安全提示

- 不要提交应用 `.env` 或 `install/milvus/.env`。
- 生产环境必须修改 MinIO 默认账号和密码。
- 模型和虚拟环境不会被打入 Docker 构建上下文。
- Docker 镜像构建已通过 `.dockerignore` 排除 `.env`，避免 API Key 进入镜像。

## 变更记录

参见 [CHANGELOG.md](./CHANGELOG.md)。
