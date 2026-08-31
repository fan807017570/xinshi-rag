#!/usr/bin/env bash
# =============================================================================
# build-macos.sh — macOS 下构建 xinshi-rag Docker 镜像
#
# 用法与 build.sh 相同：
#   bash rag/docker-install/build-macos.sh
#   bash rag/docker-install/build-macos.sh 1.0.0
#   REGISTRY=registry.example.com/xinshi \
#     bash rag/docker-install/build-macos.sh 1.0.0 --push
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未找到 docker，请先执行 bash install/macos/install-deps.sh --install --docker-only" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "错误：Docker Desktop 未运行。" >&2
  exit 1
fi

export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DOCKER_DEFAULT_PLATFORM="${DOCKER_DEFAULT_PLATFORM:-$DOCKER_PLATFORM}"
echo "==> Docker 平台：$DOCKER_PLATFORM"

exec bash "$SCRIPT_DIR/build.sh" "$@"
