# 安装与基础设施

`install` 目录负责 macOS / Ubuntu 基础依赖和 Milvus 基础设施。RAG 应用的 Docker 部署脚本位于 `rag/docker-install/`，原生构建脚本位于 `rag/build/`。

## macOS 首次安装

Docker 部署：

```bash
bash install/macos/install-deps.sh --install --docker-only
bash install/macos/deploy-components.sh up
```

应用 Docker 部署：

```bash
cp rag/.env.example rag/.env
# 编辑 rag/.env，填写 DEEPSEEK_API_KEY
bash rag/docker-install/build-macos.sh
bash rag/docker-install/deploy-macos.sh
```

应用原生部署：

```bash
bash install/macos/install-deps.sh --install
bash rag/build/build-macos.sh
bash rag/build/run-macos.sh
```

Apple Silicon 默认使用 `linux/amd64` 运行 Docker 基础组件和应用；如确认镜像支持 ARM64，可设置 `DOCKER_PLATFORM=linux/arm64`。

应用运行参数位于 `config/application.ini`。Docker 会挂载该配置目录；修改后执行下面的命令即可生效，不需要重新构建镜像：

```bash
bash rag/docker-install/deploy-macos.sh --restart
```

## Ubuntu 首次安装

```bash
bash install/install-docker.sh
bash install/install-ubuntu-deps.sh
```

如果只使用 Docker 部署应用，第二个脚本不是必需的；如果使用 Ubuntu 原生运行方式，建议两个脚本都执行。

## 启动 Milvus

```bash
bash install/milvus/install.sh up
```

Milvus、etcd 和 MinIO 使用独立 Compose 管理，共享网络名为 `xinshi-net`。首次执行会从 `.env.example` 创建 `install/milvus/.env`，默认沿用旧一体化 Compose 的项目名和数据卷名，便于复用已有 Milvus 数据；可按需修改端口和卷名。

## 应用部署方式

Docker 方式仍使用：

```bash
cp rag/.env.example rag/.env
# 编辑 rag/.env，至少填写 DEEPSEEK_API_KEY
bash rag/docker-install/build.sh
bash rag/docker-install/deploy.sh
```

Ubuntu 原生方式使用：

```bash
cp rag/.env.example rag/.env
# 在 config/application.ini 修改模型等运行参数，在 rag/.env 修改 API Key
bash rag/build/build-ubuntu.sh
bash rag/build/run-ubuntu.sh
```

原生方式默认连接 `127.0.0.1:19531`，因此仍需先启动 `install/milvus/install.sh`；如果 Milvus 使用了其他端口，请同步修改 `config/application.ini`。
