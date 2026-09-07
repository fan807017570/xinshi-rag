# Changelog

本文档记录 `xinshi-rag` 项目的重要变更。

## [Unreleased] - 2026-08-29

### 新增

- 新增网页非流式聊天接口 `POST /api/chat/completions`，并将 `index.html` 切换到该接口。
- 将成绩查询意图和普通 RAG 合并到唯一的 `xinshi-unified-chat` LangGraph；网页同步接口与微信公众号入口共用该编排。
- 网页端支持成绩查询参数补充，参数完整后以纯文本展示经过 URL 编码的固定 H5 入口，不生成超链接。
- 成绩意图调整为三级识别：规则明确查询时直接返回链接，规则明确非查询时直接进入 RAG，仅不确定表达调用 LLM。
- 成绩查询命中后使用独立零温度 LLM 提取姓名、时间、考试类型和学科，缺少必填参数时追问，完整后拼接到固定 H5 URL。
- 聊天响应新增稳定 `document_id`、相对 `source`、检索/重排分数及分层检索指标。
- 新增 `/api/docs/index` 索引覆盖率接口和 JSONL RAG 分层评测工具。
- 新增外部运行配置 `config/application.ini`，集中管理原 `config.py` 中的 Milvus、模型、重排、检索和会话参数。
- 配置支持 `native` / `docker` profile，以及环境变量兼容覆盖。
- Docker 应用只读挂载宿主机 `config/` 目录，修改配置后只需重启应用，无需重新构建镜像。
- 新增 macOS 部署脚本：
  - `install/macos/install-deps.sh`：检查/安装 Homebrew、Python 3.11 和 Docker Desktop。
  - `install/macos/deploy-components.sh`：部署 Milvus、etcd 和 MinIO 基础组件。
  - `rag/docker-install/build-macos.sh`、`deploy-macos.sh`：构建和部署应用 Docker。
  - `rag/build/requirements-macos.txt`、`build-macos.sh`、`run-macos.sh`：原生 macOS 应用部署。

- 将 RAG 项目迁移为独立的 `xinshi-rag` 项目。
- 新增 `install/install-docker.sh`，用于在 Ubuntu 上安装 Docker Engine、Buildx 和 Docker Compose Plugin。
- 新增 `install/install-ubuntu-deps.sh`，用于安装 Ubuntu 原生运行所需的 Python、venv 和编译依赖。
- 新增独立 Milvus 基础设施目录 `install/milvus/`：
  - `docker-compose.yml`：独立编排 Milvus、etcd 和 MinIO。
  - `.env.example`：配置端口、共享网络、Compose 项目名和数据卷名。
  - `install.sh`：支持 `up`、`down`、`restart`、`status` 和 `logs` 操作。
- 新增 Ubuntu 原生应用构建脚本 `rag/build/build-ubuntu.sh`。
- 新增 Ubuntu 原生应用启动脚本 `rag/build/run-ubuntu.sh`。
- 新增 `requirements-common.txt`，统一 Docker、本地和 Ubuntu 构建的公共 Python 依赖。
- 新增 `rag/build/requirements.txt`，使用 PyTorch 官方 CPU 软件源。
- 新增 `.dockerignore`，避免将虚拟环境、Git 数据、缓存、模型和 `.env` 密钥打入 Docker 镜像。
- 新增安装说明 `install/README.md` 和完整项目部署说明 `README.md`。

### 调整

- 将原来的一体化 Docker Compose 拆分为两部分：
  - `rag/docker-install/docker-compose.yml` 只负责 `xinshi-rag` 应用。
  - `install/milvus/docker-compose.yml` 负责 Milvus、etcd 和 MinIO。
- 将应用 Dockerfile、Compose、镜像构建和部署脚本统一移动到 `rag/docker-install/`。
- 将 Ubuntu 原生构建、运行脚本和 CPU 依赖清单统一移动到 `rag/build/`。
- 删除冗余的 `arg/` 目录，将应用源码从 `arg/xinshi/` 提升到根目录 `rag/`，Python 包名同步调整为 `rag`。
- Docker 应用通过外部共享网络 `xinshi-net` 访问 `milvus-standalone:19530`。
- Ubuntu 原生应用默认通过宿主机地址 `127.0.0.1:19531` 访问 Milvus。
- macOS Apple Silicon 默认使用 `linux/amd64` 兼容模式运行 Docker 组件；可通过 `DOCKER_PLATFORM` 覆盖。
- Milvus 基础设施沿用旧 Compose 项目名 `xinshi` 和旧数据卷名，便于复用已有数据。
- 保留 Docker 镜像构建和部署能力，入口调整为 `rag/docker-install/build.sh` 与 `rag/docker-install/deploy.sh`。
- `deploy.sh` 改为兼容 `docker compose` 和旧版 `docker-compose` 命令，并在启动应用前检查共享网络。
- Dockerfile 改为安装公共依赖和 CPU-only PyTorch。
- 默认依赖、Docker 镜像和 Ubuntu 原生构建全部统一为 CPU-only PyTorch；服务器无需安装 CUDA、cuDNN 或 NVIDIA 驱动。
- 将模型默认路径从开发机器的绝对路径调整为项目内相对路径：
  - `./rag/models/bge-base-zh-v1.5`
  - `./rag/models/bge-reranker-large`
- 更新应用 `.env.example`，区分 Docker 容器内配置和 Ubuntu 原生运行配置。

### 修复

- LLM 配置与请求改为严格失败：缺少 Key、参数错误、客户端初始化失败或远程请求失败时返回明确错误，不再生成本地兜底回答。
- 移除源码和 `.env.example` 中的硬编码 DeepSeek 凭据，健康检查新增非敏感 LLM backend/model 状态。
- Milvus 启用时连接失败直接终止启动，不再静默降级到本地字符串检索。
- Reranker 候选上限调整为 60，并在配置不足时拒绝启动，避免召回结果被静默截断。
- 索引重建后校验实体来源覆盖率，并统一 native/Docker 环境下的相对来源路径。
- 修复 `langchain-community 0.4.x` 不再从 `vectorstores` 顶层导出 `Milvus` 导致的启动失败。
- `ingest.py` 改为直接读取 Markdown，不再依赖已停止维护的 community 文档加载器。
- 将 Milvus 旧适配器隔离到兼容模块，并把 PyMilvus 限定在与 Milvus 2.4.x 匹配的版本范围。
- macOS 原生构建优先选择 Apple Silicon arm64 Python，并强制安装 `cryptography` wheel，避免旧 Cargo 源码构建失败。
- setuptools 限制在 81 以下，以兼容 PyMilvus 2.4 对 `pkg_resources` 的运行时依赖并避免弃用警告。

### 部署方式

- Docker 部署：构建 `xinshi-rag` 镜像，并通过独立 Compose 启动应用。
- macOS / Ubuntu 原生部署：在项目根目录创建 `.venv`，安装 CPU 依赖后直接通过 Uvicorn 启动应用。
- 两种应用部署方式都可以复用 `install/milvus/` 管理的 Milvus 基础设施。

### 验证

- 通过全部 Shell 脚本的 `bash -n` 语法检查。
- 通过全部 Python 文件的 AST 语法检查。
- 通过应用和 Milvus 两份 Docker Compose 配置解析检查。
- 确认应用 Compose 仅包含 `xinshi-rag` 服务。
- 确认基础设施 Compose 包含 `etcd`、`minio` 和 `milvus-standalone` 服务。
- 确认所有安装、构建和启动脚本均具有可执行权限。

### 注意事项

- 未在本次变更中实际执行 `apt` 安装、Python 依赖下载、Docker 镜像构建或容器启动。
- 首次部署前必须配置有效的 `DEEPSEEK_API_KEY` 和可用的模型目录。
